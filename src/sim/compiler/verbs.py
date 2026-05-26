"""Verb-specific compile functions."""

from __future__ import annotations

import hashlib
import logging
import random
from typing import Any, Optional

logger = logging.getLogger(__name__)

from ..relational import (
    get_entity_faction_ids,
    get_faction_tension,
)
from ..schemas import (
    ActionType,
    ActionZone,
    AlertnessLevel,
    ConsentState,
    Coord,
    EdgeKind,
    EmotionalState,
    EntityId,
    EntityState,
    EquipSlot,
    FacingDirection,
    FACING_VECTORS,
    GroundingResult,
    IntentBlock,
    MalformedActionError,
    ObjectId,
    ObjectState,
    ProjectionFirewallError,
    RejectionReason,
    SemanticAction,
    SemanticProjection,
    SpatialGrid,
    StyleBlock,
    TerrainType,
    Transition,
    TransitionKind,
    TransitionProposal,
    ValidationResult,
    VerbTemplate,
    WorldState,
    coord_key,
)
from ..spatial import (
    entities_visible_from,
    find_path,
    is_reachable,
    visible_from,
)
from typing import Callable

from .common import *
from .beliefs import build_belief_transitions
from .proposals import _substitute_proposal, _validate_proposals

def _compile_move(
    action: SemanticAction,
    world: WorldState,
) -> ValidationResult:
    try:
        grid = world.grid_for_entity(action.actor)
    except KeyError:
        grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.ACTOR_NOT_FOUND,
            rejection_detail=f"Actor '{action.actor}' not found in grid.",
        )

    # Pre-resolve relative-motion keywords ("forward", "back", "left", "right")
    # into absolute Coord targets before the main target-type dispatch.
    if isinstance(action.target, str):
        rel_coord = _resolve_relative_move(str(action.target), actor)
        if rel_coord is not None:
            action = action.model_copy(update={"target": rel_coord})

    if isinstance(action.target, str):
        # EntityId is NewType(str) — move toward the named entity.
        # Try exact entity-id match first; fall back to name matching so
        # that LM-generated actions referencing entities by name (e.g.
        # "Guard", "player") rather than canonical ID still resolve.
        target_entity = _resolve_entity_target(grid, action.target)
        if target_entity is None:
            return ValidationResult(
                valid=False,
                rejection_reason=RejectionReason.TARGET_NOT_FOUND,
                rejection_detail=f"Move-to-entity target '{action.target}' not found in grid.",
            )
        # Pick the adjacent tile closest to the actor
        pos = target_entity.position
        candidates = [
            Coord(x=pos.x + dx, y=pos.y + dy)
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0))
        ]
        passable_candidates = [
            c for c in candidates
            if is_reachable(grid, actor.position, c)
        ]
        if not passable_candidates:
            return ValidationResult(
                valid=False,
                rejection_reason=RejectionReason.PATH_INACCESSIBLE,
                rejection_detail=f"No passable tile adjacent to '{action.target}'.",
            )
        # Closest by manhattan distance from actor
        destination: Coord = min(
            passable_candidates,
            key=lambda c: c.manhattan(actor.position),
        )
    elif isinstance(action.target, Coord):
        destination: Coord = action.target  # type: ignore[no-redef]
        if not grid.is_in_bounds(destination):
            return ValidationResult(
                valid=False,
                rejection_reason=RejectionReason.PATH_INACCESSIBLE,
                rejection_detail=f"Destination {destination} is out of bounds.",
            )
        # Speed-clamp: clamp movement to the entity's speed attribute.
        # Find the full path to the destination and take only the first N
        # tiles, so long-distance moves are broken into multiple turns.
        speed = actor.attributes.get("speed", 4)
        if actor.position != destination:
            path = find_path(grid, actor.position, destination, max_steps=512)
            if path is None:
                return ValidationResult(
                    valid=False,
                    rejection_reason=RejectionReason.PATH_INACCESSIBLE,
                    rejection_detail=(
                        f"No passable path from {actor.position} to {destination}."
                    ),
                )
            # path is a list of Coords NOT including the start position.
            destination = path[min(speed - 1, len(path) - 1)]
    else:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.MALFORMED_ACTION,
            rejection_detail="MOVE action requires a Coord or EntityId target.",
        )

    # Encumbrance gate: over-capacity entities cannot move at all.
    enc = _encumbrance_ratio(actor, grid)
    if enc > 1.0:
        carried = current_carry_weight(actor, grid)
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=(
                f"Over-encumbered: carrying {carried:.1f} kg "
                f"(capacity {actor.carry_capacity:.1f} kg). "
                f"Drop or store items before moving."
            ),
        )

    # Check for portals at the destination — generate REGION_TRANSIT instead
    # of a plain ENTITY_MOVED when stepping onto a portal tile.
    portal = world.portal_at(actor.region_id, destination)
    if portal is not None:
        # Key-item gate
        if portal.requires_key and portal.requires_key not in actor.inventory:
            obj = grid.objects.get(portal.requires_key)
            key_name = obj.name if obj else str(portal.requires_key)
            return ValidationResult(
                valid=False,
                rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
                rejection_detail=f"The passage is locked. You need: {key_name}.",
            )
        arrival = portal.to_coord
        return ValidationResult(
            valid=True,
            concrete_transitions=[
                Transition(
                    kind=TransitionKind.REGION_TRANSIT,
                    payload={
                        "entity_id": action.actor,
                        "from_region": actor.region_id,
                        "to_region": portal.to_region,
                        "arrival": {"x": arrival.x, "y": arrival.y},
                    },
                )
            ],
            plausibility=1.0,
        )

    move_transition = Transition(
        kind=TransitionKind.ENTITY_MOVED,
        payload={
            "entity_id": action.actor,
            "from": {"x": actor.position.x, "y": actor.position.y},
            "to": {"x": destination.x, "y": destination.y},
        },
    )
    from ..hazard_avoidance import movement_hazard_transitions

    hazard_extras = movement_hazard_transitions(world, action.actor, destination)
    extras = _validate_proposals(action.proposed_effects or [], world, action)
    return ValidationResult(
        valid=True,
        concrete_transitions=[move_transition] + hazard_extras + extras,
        plausibility=1.0,
    )


def _compile_social(
    action: SemanticAction,
    world: WorldState,
) -> ValidationResult:
    """
    Generic resolver for social/verbal actions:
    intimidate, persuade, deceive, bribe, threaten, speak, ask.
    """
    grid = world.spatial
    graph = world.relational

    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.ACTOR_NOT_FOUND,
        )

    if action.target is None or isinstance(action.target, Coord):
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.MALFORMED_ACTION,
            rejection_detail="Social actions require an EntityId target.",
        )

    target = _resolve_entity_target(grid, str(action.target))
    if target is None:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.TARGET_NOT_FOUND,
            rejection_detail=f"Target entity '{action.target}' not found in grid.",
        )
    target_id = target.entity_id

    # Check line of sight
    visible_from_actor = visible_from(grid, actor.position, actor.sight_range)
    if target.position not in visible_from_actor:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.ENTITY_NOT_VISIBLE,
            rejection_detail=(
                f"Target '{target_id}' is not visible from actor '{action.actor}'."
            ),
        )

    # Attribute contest
    attr_map = {
        ActionType.INTIMIDATE: ("intimidation", "perception"),
        ActionType.PERSUADE: ("persuasion", "perception"),
        ActionType.DECEIVE: ("deception", "perception"),
        ActionType.BRIBE: ("persuasion", "perception"),
        ActionType.THREATEN: ("intimidation", "perception"),
        ActionType.SPEAK: (None, None),
        ActionType.ASK: (None, None),
    }
    actor_attr_name, target_attr_name = attr_map.get(
        action.verb, (None, None)
    )

    transitions: list[Transition] = []
    plausibility = 1.0

    if actor_attr_name and target_attr_name:
        actor_score = actor.attributes.get(actor_attr_name, 50)
        target_score = target.attributes.get(target_attr_name, 50)

        # Modifiers
        if target.alertness == AlertnessLevel.HIGH:
            target_score += 15
        elif target.alertness == AlertnessLevel.UNAWARE:
            target_score -= 10
        if action.style.aggression > 70:
            actor_score += 10
        if action.constraints.avoid_witnesses:
            # Check witnesses
            witnesses = [
                e for e in grid.entities.values()
                if e.entity_id not in (action.actor, target_id)
                and e.position in visible_from(grid, actor.position, actor.sight_range)
            ]
            if witnesses and action.verb == ActionType.INTIMIDATE:
                target_score += 10  # harder to intimidate publicly

        seed = f"{action.action_id}_{world.tick}"
        success = _contest_roll(actor_score, target_score, seed)
        plausibility = min(1.0, max(0.0, (actor_score - target_score + 20) / 40))

        if success:
            transitions.append(
                Transition(
                    kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED,
                    payload={
                        "entity_id": target_id,
                        "from": target.emotional_state.value,
                        "to": _outcome_state(action.verb, success=True).value,
                        "cause": action.verb,
                        "actor": action.actor,
                    },
                )
            )
            # Increase target alertness on threatening interactions
            if action.verb in (
                ActionType.INTIMIDATE,
                ActionType.THREATEN,
            ):
                transitions.append(
                    Transition(
                        kind=TransitionKind.ENTITY_ALERTNESS_CHANGED,
                        payload={
                            "entity_id": target_id,
                            "from": target.alertness.value,
                            "to": AlertnessLevel.MEDIUM.value,
                        },
                    )
                )
        else:
            transitions.append(
                Transition(
                    kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED,
                    payload={
                        "entity_id": target_id,
                        "from": target.emotional_state.value,
                        "to": _outcome_state(action.verb, success=False).value,
                        "cause": action.verb,
                        "actor": action.actor,
                    },
                )
            )
            if action.verb == ActionType.INTIMIDATE:
                transitions.append(
                    Transition(
                        kind=TransitionKind.ENTITY_ALERTNESS_CHANGED,
                        payload={
                            "entity_id": target_id,
                            "from": target.alertness.value,
                            "to": AlertnessLevel.HIGH.value,
                        },
                    )
                )
    else:
        # Pure speech — always succeeds, target may respond
        transitions.append(
            Transition(
                kind=TransitionKind.DIALOGUE_SPOKEN,
                payload={
                    "actor": action.actor,
                    "target": target_id,
                    "intent": action.intent.model_dump(),
                },
        )
    )

    # ── Gossip/Belief propagation on social speech ─────────────────────────
    # If the actor holds any beliefs, they might share/gossip about one of them.
    # We support this for speak, ask, deceive, persuade, and gossip.
    if action.verb in (ActionType.SPEAK, ActionType.ASK, ActionType.DECEIVE, ActionType.PERSUADE, "gossip", "whisper"):
        from ..schemas import EdgeKind
        actor_id_str = str(action.actor)
        held_beliefs = []
        for tgt_node_id, edges in world.relational.edges.get(actor_id_str, {}).items():
            for edge in edges:
                if edge.kind == EdgeKind.BELIEVES_CLAIM:
                    held_beliefs.append((tgt_node_id, edge))

        if held_beliefs:
            # Look for keyword matches in intent text if present
            intent_text = ""
            if action.intent:
                text_parts = []
                if hasattr(action.intent, "manner") and action.intent.manner:
                    text_parts.append(action.intent.manner)
                if hasattr(action.intent, "rationale") and action.intent.rationale:
                    text_parts.append(action.intent.rationale)
                intent_text = " ".join(text_parts).lower()

            selected_tgt = None
            selected_edge = None

            if intent_text:
                best_overlap = 0
                for tgt_node_id, edge in held_beliefs:
                    claim = edge.meta.get("claim", "").lower()
                    words = [w for w in claim.split() if len(w) >= 4]
                    overlap = sum(1 for w in words if w in intent_text)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        selected_tgt = tgt_node_id
                        selected_edge = edge

            # Fallback: if action verb is literally gossip/whisper, or text talks about rumor/secret/gossip
            if not selected_edge and (action.verb in ("gossip", "whisper") or any(k in intent_text for k in ("gossip", "rumor", "rumour", "secret"))):
                # Pick the highest weight (most confident) belief
                held_beliefs.sort(key=lambda x: x[1].weight, reverse=True)
                selected_tgt, selected_edge = held_beliefs[0]

            if selected_edge:
                claim_text = selected_edge.meta.get("claim", "")
                new_fidelity = selected_edge.weight * 0.8
                transitions.append(
                    Transition(
                        kind=TransitionKind.BELIEF_PROPAGATED,
                        payload={
                            "from_entity": actor_id_str,
                            "to_entity": target_id,
                            "belief": claim_text,
                            "original_source": selected_tgt,
                            "fidelity": round(new_fidelity, 3),
                        }
                    )
                )

    extras = _validate_proposals(action.proposed_effects or [], world, action)
    return ValidationResult(
        valid=True,
        concrete_transitions=transitions + extras,
        plausibility=plausibility,
    )


def _outcome_state(
    verb: str, *, success: bool
) -> EmotionalState:
    if success:
        return {
            ActionType.INTIMIDATE: EmotionalState.FEARFUL,
            ActionType.PERSUADE: EmotionalState.FRIENDLY,
            ActionType.DECEIVE: EmotionalState.NEUTRAL,
            ActionType.BRIBE: EmotionalState.FRIENDLY,
            ActionType.THREATEN: EmotionalState.FEARFUL,
        }.get(verb, EmotionalState.NEUTRAL)
    else:
        return {
            ActionType.INTIMIDATE: EmotionalState.ANGRY,
            ActionType.PERSUADE: EmotionalState.SUSPICIOUS,
            ActionType.DECEIVE: EmotionalState.SUSPICIOUS,
            ActionType.BRIBE: EmotionalState.ANGRY,
            ActionType.THREATEN: EmotionalState.HOSTILE,
        }.get(verb, EmotionalState.SUSPICIOUS)


def _effective_attack_range(actor: "EntityState", grid) -> tuple[int, str]:
    """
    Return (range_in_tiles, description) for an attack by this entity.

    Range rules (1 tile ≈ 1 metre):
      armed=True              → ranged weapon, 8 tiles (~8m)
      has items in inventory  → can throw, 6 tiles (~6m)
      otherwise               → unarmed melee, 1 tile (adjacent)
    """
    if actor.armed:
        return 8, "ranged weapon"
    if actor.inventory:
        return 6, "thrown item"
    return 1, "unarmed melee"


def _get_weapon_damage(actor: "EntityState", grid: "SpatialGrid") -> int:
    """Return the damage bonus from the actor's equipped weapon (or best in inventory)."""
    if actor.equipped_weapon:
        obj = grid.objects.get(actor.equipped_weapon)
        if obj:
            return obj.attributes.get("damage", 0)
    # Fall back to the first damage-bearing item in inventory.
    for oid in actor.inventory:
        obj = grid.objects.get(oid)
        if obj and obj.attributes.get("damage", 0) > 0:
            return obj.attributes["damage"]
    return 0


def _get_armor_protection(target: "EntityState", grid: "SpatialGrid") -> int:
    """Return the protection value from the target's equipped armor."""
    if target.equipped_armor:
        obj = grid.objects.get(target.equipped_armor)
        if obj:
            return obj.attributes.get("protection", 0)
    # Fall back to any protection-bearing item in inventory.
    for oid in target.inventory:
        obj = grid.objects.get(oid)
        if obj and obj.attributes.get("protection", 0) > 0:
            return obj.attributes["protection"]
    return 0


def _compile_attack(
    action: SemanticAction,
    world: WorldState,
) -> ValidationResult:
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(
            valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND
        )
    if action.target is None or isinstance(action.target, Coord):
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.MALFORMED_ACTION,
            rejection_detail="ATTACK requires an EntityId target.",
        )
    target = _resolve_entity_target(grid, str(action.target))
    if target is None:
        return ValidationResult(
            valid=False, rejection_reason=RejectionReason.TARGET_NOT_FOUND
        )
    target_id = target.entity_id
    if target.position not in visible_from(grid, actor.position, actor.sight_range):
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.ENTITY_NOT_VISIBLE,
        )
    dist = actor.position.manhattan(target.position)
    weapon_range, range_label = _effective_attack_range(actor, grid)
    if dist > weapon_range:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.TARGET_OUT_OF_RANGE,
            rejection_detail=(
                f"Target is {dist} tiles away; {range_label} range is {weapon_range}."
            ),
        )
    base_damage = actor.attributes.get("strength", 50) // 10
    # Weapon bonus from equipped item or any damage-bearing item in inventory.
    weapon_bonus = _get_weapon_damage(actor, grid)
    # Armor reduction on the target.
    armor_reduction = _get_armor_protection(target, grid)
    from ..combat_tactics import (
        cover_protection,
        range_damage_multiplier,
        range_zone,
        wear_from_attack,
    )

    cover_reduction = cover_protection(grid, actor.position, target.position)
    seed = f"{action.action_id}_{world.tick}_dmg"
    noise = int(_seeded_float(seed) * 6)
    raw_damage = base_damage + weapon_bonus - armor_reduction - cover_reduction + noise
    range_mult = range_damage_multiplier(dist, range_label)
    damage = max(1, int(raw_damage * range_mult))

    # Back-attack bonus: +50% damage when striking outside the target's FOV cone.
    # This rewards flanking and stealth play.
    from ..spatial import _in_facing_cone, _cone_cos
    dx = actor.position.x - target.position.x
    dy = actor.position.y - target.position.y
    target_half_cos = _cone_cos(target.fov_degrees)
    is_backstab = not _in_facing_cone(dx, dy, target.facing, target_half_cos)
    if is_backstab:
        damage = int(damage * 1.5)
        # Encode backstab flag in the health-change payload for the narrator.

    noise_level = 70 if weapon_bonus > 0 else 50  # louder with a weapon
    transitions = [
        Transition(
            kind=TransitionKind.ENTITY_HEALTH_CHANGED,
            payload={
                "entity_id": target_id,
                "delta": -damage,
                "cause": "attack",
                "actor": action.actor,
                "backstab": is_backstab,
            },
        ),
        Transition(
            kind=TransitionKind.ENTITY_ALERTNESS_CHANGED,
            payload={
                "entity_id": target_id,
                "from": target.alertness.value,
                "to": AlertnessLevel.COMBAT.value,
            },
        ),
        Transition(
            kind=TransitionKind.NOISE_EVENT,
            payload={
                "source_id": action.actor,
                "source_pos": {"x": actor.position.x, "y": actor.position.y},
                "noise_level": noise_level,
            },
        ),
    ]
    # If attack used a throw, consume one item from inventory
    if not actor.armed and actor.inventory:
        thrown_item = actor.inventory[0]
        transitions.append(
            Transition(
                kind=TransitionKind.ITEM_TRANSFERRED,
                payload={
                    "object_id": thrown_item,
                    "from_entity": action.actor,
                    "to_entity": target_id,
                },
            )
        )
    if damage >= 3:
        from ..contamination import bleed_at_entity
        transitions.extend(
            bleed_at_entity(world, target, damage=damage, cause="attack")
        )
    transitions.extend(wear_from_attack(actor, target, grid, damage))
    if cover_reduction > 0:
        transitions[0].payload["cover_reduction"] = cover_reduction
        transitions[0].payload["range_zone"] = range_zone(dist)
    return ValidationResult(
        valid=True,
        concrete_transitions=transitions,
        plausibility=1.0,
    )


# ---------------------------------------------------------------------------
# Item-name fuzzy finder
# ---------------------------------------------------------------------------

def _find_item_by_name(
    name_hint: str,
    actor: "EntityState",
    grid: "SpatialGrid",
    *,
    also_ground: bool = False,
    ground_range: int = 1,
) -> "Optional[ObjectState]":
    """
    Find an ObjectState in the actor's inventory (or nearby ground) that best
    matches ``name_hint``.  The matching is fuzzy: exact match > substring >
    shared-word count.

    If ``also_ground`` is True, items lying on the floor within ``ground_range``
    tiles are also considered.

    Returns None if nothing plausible is found.
    """
    if not name_hint:
        return None
    nl = name_hint.lower().strip()
    candidates: list[ObjectState] = []
    for oid in actor.inventory:
        obj = grid.objects.get(oid)
        if obj:
            candidates.append(obj)
    if also_ground:
        for obj in grid.objects.values():
            if obj.position is not None and obj.owner is None:
                if actor.position.manhattan(obj.position) <= ground_range:
                    candidates.append(obj)

    best: Optional[ObjectState] = None
    best_score = -1
    for obj in candidates:
        ol = obj.name.lower()
        if ol == nl:
            return obj  # exact
        score = 0
        if nl in ol or ol in nl:
            score = 10
        shared = len(set(nl.split()) & set(ol.split()))
        score += shared
        if score > best_score:
            best_score = score
            best = obj
    return best if best_score >= 0 else None


def _item_physics_on_hit(
    thrown_obj: "ObjectState",
    target: "EntityState",
    grid: "SpatialGrid",
) -> "list[Transition]":
    """
    Secondary physics transitions triggered when a thrown item hits an entity.

    Rules:
      on_fire + biological target  → add on_fire tag (ignite target)
      corrosive (acid)             → deal acid contact damage + burned tag
      toxic (poison)               → add poisoned tag to target
      fire_suppressant + on_fire   → extinguish target's fire
    """
    transitions: list[Transition] = []
    eid = str(target.entity_id)

    item_tags = set(thrown_obj.tags)
    target_tags = set(target.tags)

    # Burning item ignites biological targets
    if "on_fire" in item_tags and "biological" in target_tags:
        if "on_fire" not in target_tags:
            transitions.append(Transition(
                kind=TransitionKind.ENTITY_CONDITION_CHANGED,
                payload={"entity_id": eid, "condition": "tag_on_fire",
                         "ticks": 9999, "_tag_add": "on_fire"},
            ))

    # Acid flask — extra burn damage
    if "corrosive" in item_tags:
        acid_dmg = int(thrown_obj.attributes.get("damage", 8))
        transitions.append(Transition(
            kind=TransitionKind.ENTITY_HEALTH_CHANGED,
            payload={"entity_id": eid, "delta": -acid_dmg,
                     "cause": "acid_contact", "actor": eid},
        ))
        if "burned" not in target_tags:
            transitions.append(Transition(
                kind=TransitionKind.ENTITY_CONDITION_CHANGED,
                payload={"entity_id": eid, "condition": "tag_burned",
                         "ticks": 9999, "_tag_add": "burned"},
            ))

    # Poison flask
    if "toxic" in item_tags:
        if "poisoned" not in target_tags:
            transitions.append(Transition(
                kind=TransitionKind.ENTITY_CONDITION_CHANGED,
                payload={"entity_id": eid, "condition": "tag_poisoned",
                         "ticks": 9999, "_tag_add": "poisoned"},
            ))

    # Water / fire-suppressant extinguishes fire on target
    if "fire_suppressant" in item_tags and "on_fire" in target_tags:
        transitions.append(Transition(
            kind=TransitionKind.ENTITY_CONDITION_CHANGED,
            payload={"entity_id": eid, "condition": "tag_on_fire",
                     "ticks": 0, "_tag_remove": "on_fire"},
        ))
        transitions.append(Transition(
            kind=TransitionKind.ENTITY_HEALTH_CHANGED,
            payload={"entity_id": eid, "delta": 0,
                     "prop_delta": {"temperature": -80},
                     "cause": "water_doused", "actor": eid},
        ))

    return transitions


def _compile_throw(action: SemanticAction, world: WorldState) -> ValidationResult:
    """
    THROW kernel — improved version.

    Differences from the old stub that delegated to _compile_attack:

    1. Finds the specific named item from the actor's inventory (not inventory[0]).
    2. Applies the item's attributes.damage as the primary damage source.
    3. Item lands AT the target's position (not in their inventory).
    4. Applies physics side-effects based on the item's tags (acid, fire, poison, water).
    """
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(
            valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND
        )
    if action.target is None or isinstance(action.target, Coord):
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.MALFORMED_ACTION,
            rejection_detail="THROW requires an entity target.",
        )
    target = _resolve_entity_target(grid, str(action.target))
    if target is None:
        return ValidationResult(
            valid=False, rejection_reason=RejectionReason.TARGET_NOT_FOUND
        )

    # Range — throwing requires line-of-sight
    if target.position not in visible_from(grid, actor.position, actor.sight_range):
        return ValidationResult(
            valid=False, rejection_reason=RejectionReason.ENTITY_NOT_VISIBLE
        )
    dist = actor.position.manhattan(target.position)
    throw_range = 7  # further than melee, closer than a bow
    if dist > throw_range:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.TARGET_OUT_OF_RANGE,
            rejection_detail=f"Target is {dist} tiles away; max throw range is {throw_range}.",
        )

    # Find item to throw — prefer named item from intent, fall back to first throwable
    item_hint = ""
    if action.intent:
        item_hint = action.intent.rationale or action.intent.manner or ""
    thrown_obj: Optional[ObjectState] = None
    if item_hint:
        thrown_obj = _find_item_by_name(item_hint, actor, grid)
    if thrown_obj is None:
        # Prefer items tagged "throwable"
        for oid in actor.inventory:
            obj = grid.objects.get(oid)
            if obj and "throwable" in obj.tags:
                thrown_obj = obj
                break
    if thrown_obj is None and actor.inventory:
        thrown_obj = grid.objects.get(actor.inventory[0])
    if thrown_obj is None:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail="You have nothing to throw.",
        )

    # Damage: item's damage attribute; if item has no damage, use a small default
    item_damage = int(thrown_obj.attributes.get("damage", 5))
    armor_reduction = _get_armor_protection(target, grid)
    seed = f"{action.action_id}_{world.tick}_throw"
    noise = int(_seeded_float(seed) * 4)
    damage = max(1, item_damage - armor_reduction + noise)

    transitions: list[Transition] = [
        Transition(
            kind=TransitionKind.ENTITY_HEALTH_CHANGED,
            payload={
                "entity_id": target.entity_id,
                "delta": -damage,
                "cause": "thrown_item",
                "actor": action.actor,
                "thrown_item_name": thrown_obj.name,
            },
        ),
        Transition(
            kind=TransitionKind.ENTITY_ALERTNESS_CHANGED,
            payload={
                "entity_id": target.entity_id,
                "from": target.alertness.value,
                "to": AlertnessLevel.COMBAT.value,
            },
        ),
        Transition(
            kind=TransitionKind.NOISE_EVENT,
            payload={
                "source_id": action.actor,
                "source_pos": {"x": actor.position.x, "y": actor.position.y},
                "noise_level": 60,
            },
        ),
        # Item lands at the target's feet — NOT in their inventory
        Transition(
            kind=TransitionKind.ITEM_TRANSFERRED,
            payload={
                "object_id": thrown_obj.object_id,
                "from_entity": action.actor,
                "to_entity": None,
                "to_position": {"x": target.position.x, "y": target.position.y},
            },
        ),
    ]

    # Physics side-effects (fire, acid, poison, water)
    transitions.extend(_item_physics_on_hit(thrown_obj, target, grid))

    return ValidationResult(valid=True, concrete_transitions=transitions, plausibility=1.0)


# ---------------------------------------------------------------------------
# DROP
# ---------------------------------------------------------------------------

def _compile_drop(action: SemanticAction, world: WorldState) -> ValidationResult:
    """
    DROP kernel — remove an item from inventory and place it at the actor's feet.

    The item name / hint is taken from action.intent.rationale.
    If no name is given and the actor carries something, the first inventory
    item is dropped.
    """
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(
            valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND
        )

    item_hint = ""
    if action.intent:
        item_hint = action.intent.rationale or action.intent.manner or ""
    # If the action.target is an ObjectId, use it directly
    if action.target and not isinstance(action.target, Coord):
        target_oid = ObjectId(str(action.target))
        obj = grid.objects.get(target_oid)
        if obj and target_oid in actor.inventory:
            return ValidationResult(
                valid=True,
                concrete_transitions=[Transition(
                    kind=TransitionKind.ITEM_TRANSFERRED,
                    payload={
                        "object_id": target_oid,
                        "from_entity": action.actor,
                        "to_entity": None,
                        "to_position": {"x": actor.position.x, "y": actor.position.y},
                    },
                )],
                plausibility=1.0,
            )

    dropped = _find_item_by_name(item_hint, actor, grid) if item_hint else None
    if dropped is None:
        if not actor.inventory:
            return ValidationResult(
                valid=False,
                rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
                rejection_detail="You are not carrying anything to drop.",
            )
        dropped = grid.objects.get(actor.inventory[0])

    if dropped is None or dropped.object_id not in actor.inventory:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=f"You are not carrying '{item_hint}'.",
        )

    return ValidationResult(
        valid=True,
        concrete_transitions=[
            Transition(
                kind=TransitionKind.ITEM_TRANSFERRED,
                payload={
                    "object_id": dropped.object_id,
                    "from_entity": action.actor,
                    "to_entity": None,
                    "to_position": {"x": actor.position.x, "y": actor.position.y},
                },
            )
        ],
        plausibility=1.0,
    )


# ---------------------------------------------------------------------------
# USE
# ---------------------------------------------------------------------------

def _compile_use(action: SemanticAction, world: WorldState) -> ValidationResult:
    """
    USE kernel — use an item from inventory, optionally on a target entity.

    Dispatch rules (checked in order):
      1. consumable + attributes.heal           → ENTITY_HEALTH_CHANGED (actor heals)
      2. consumable + attributes.mana           → mana prop_delta (actor)
      3. consumable + attributes.temperature_delta + fire_suppressant
                                                → cool target / douse fire
      4. consumable + attributes.poison_dose    → add poisoned to target (or actor)
      5. light_source + target entity flammable → ignite target (add on_fire)
      6. item destroyed if consumable and successfully used
      fallback → freeform (LM proposed_effects)
    """
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(
            valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND
        )

    item_hint = ""
    if action.intent:
        item_hint = action.intent.rationale or action.intent.manner or ""

    # Find item
    item: Optional[ObjectState] = None
    if action.target and not isinstance(action.target, Coord):
        # target might be an object id
        maybe_oid = ObjectId(str(action.target))
        if maybe_oid in grid.objects and maybe_oid in actor.inventory:
            item = grid.objects[maybe_oid]
    if item is None and item_hint:
        item = _find_item_by_name(item_hint, actor, grid)
    if item is None:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=f"You don't have '{item_hint or 'that item'}' in your inventory.",
        )

    # Determine optional USE target (entity the item is used ON)
    use_target: Optional[EntityState] = None
    if action.target and not isinstance(action.target, Coord):
        use_target = _resolve_entity_target(grid, str(action.target))
        # If the resolved target is the item itself, clear it
        if use_target is not None and str(use_target.entity_id) == str(item.object_id):
            use_target = None
    if use_target is None:
        use_target = actor  # default: use on self

    actor_id = str(action.actor)
    target_id = str(use_target.entity_id)
    is_consumable = "consumable" in item.tags
    transitions: list[Transition] = []
    used = False

    # ── Healing ──────────────────────────────────────────────────────────
    heal = int(item.attributes.get("heal", 0))
    if heal > 0:
        transitions.append(Transition(
            kind=TransitionKind.ENTITY_HEALTH_CHANGED,
            payload={"entity_id": target_id, "delta": heal,
                     "cause": "consumable_heal", "actor": actor_id},
        ))
        used = True

    # ── Mana restoration ─────────────────────────────────────────────────
    mana_restore = int(item.attributes.get("mana", 0))
    if mana_restore > 0:
        transitions.append(Transition(
            kind=TransitionKind.ENTITY_HEALTH_CHANGED,
            payload={"entity_id": target_id, "delta": 0,
                     "prop_delta": {"mana": mana_restore},
                     "cause": "consumable_mana", "actor": actor_id},
        ))
        used = True

    # ── Temperature delta (water, ice water, etc.) ───────────────────────
    temp_delta = float(item.attributes.get("temperature_delta", 0))
    if temp_delta != 0:
        transitions.append(Transition(
            kind=TransitionKind.ENTITY_HEALTH_CHANGED,
            payload={"entity_id": target_id, "delta": 0,
                     "prop_delta": {"temperature": temp_delta},
                     "cause": "consumable_temp", "actor": actor_id},
        ))
        if "fire_suppressant" in item.tags and "on_fire" in use_target.tags:
            transitions.append(Transition(
                kind=TransitionKind.ENTITY_CONDITION_CHANGED,
                payload={"entity_id": target_id, "condition": "tag_on_fire",
                         "ticks": 0, "_tag_remove": "on_fire"},
            ))
        used = True

    # ── Poison dose ───────────────────────────────────────────────────────
    poison_dose = int(item.attributes.get("poison_dose", 0))
    if poison_dose > 0 and "poisoned" not in use_target.tags:
        transitions.append(Transition(
            kind=TransitionKind.ENTITY_CONDITION_CHANGED,
            payload={"entity_id": target_id, "condition": "tag_poisoned",
                     "ticks": 9999, "_tag_add": "poisoned"},
        ))
        used = True

    # ── Light source — ignite adjacent flammables ─────────────────────────
    if "light_source" in item.tags and "on_fire" in item.tags:
        # "use torch on X" — ignite the target entity if it's flammable
        if "flammable" in use_target.tags and "on_fire" not in use_target.tags:
            transitions.append(Transition(
                kind=TransitionKind.ENTITY_CONDITION_CHANGED,
                payload={"entity_id": target_id, "condition": "tag_on_fire",
                         "ticks": 9999, "_tag_add": "on_fire"},
            ))
            used = True
        elif use_target == actor:
            # Using torch with no target — just a light pulse (no-op but valid)
            used = True

    # ── Consume item if used ──────────────────────────────────────────────
    if used and is_consumable:
        transitions.append(Transition(
            kind=TransitionKind.ITEM_DESTROYED,
            payload={"object_id": item.object_id, "owner_entity": actor_id},
        ))
    elif not used:
        # Nothing to apply — fall back gracefully
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=(
                f"Using {item.name} here doesn't do anything obvious."
            ),
        )

    return ValidationResult(valid=True, concrete_transitions=transitions, plausibility=1.0)


# ---------------------------------------------------------------------------
# MIX / COMBINE
# ---------------------------------------------------------------------------

def _match_combination_rule(
    item_a: "ObjectState",
    item_b: "ObjectState",
    rules: list,
) -> "Optional[tuple[Any, bool]]":
    """
    Find the first CombinationRule whose inputs match the two items.

    Returns (rule, swapped) where swapped=True means item_a matched input_b
    and item_b matched input_a.  Returns None if no rule matches.
    """
    for rule in rules:
        a_tags = set(rule.input_a_tags)
        b_tags = set(rule.input_b_tags)
        a_match = bool(a_tags & set(item_a.tags))
        b_match = bool(b_tags & set(item_b.tags))
        if a_match and b_match:
            return rule, False
        # Try swapped
        a_match2 = bool(a_tags & set(item_b.tags))
        b_match2 = bool(b_tags & set(item_a.tags))
        if a_match2 and b_match2:
            return rule, True
    return None


def _compile_mix(action: SemanticAction, world: WorldState) -> ValidationResult:
    """
    MIX kernel — combine two items from inventory using chemistry rules.

    The player writes:
      mix <item A> with <item B>
      combine <item A> and <item B>
      use <item A> on <item B>

    The game_loop pre-processor parses those into:
      SemanticAction(verb="mix",
                     intent=IntentBundle(rationale="<name_A>",
                                         manner="<name_B>"))

    Fallback if no rule matches: reject with informative message.
    """
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(
            valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND
        )

    name_a = (action.intent.rationale or "") if action.intent else ""
    name_b = (action.intent.manner or "")   if action.intent else ""

    item_a = _find_item_by_name(name_a, actor, grid, also_ground=True)
    item_b = _find_item_by_name(name_b, actor, grid, also_ground=True)

    if item_a is None:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=f"You don't have '{name_a}'.",
        )
    if item_b is None:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=f"You don't have '{name_b}'.",
        )
    if item_a.object_id == item_b.object_id:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail="You can't mix an item with itself.",
        )

    rules = world.config.combination_rules
    match = _match_combination_rule(item_a, item_b, rules)
    if match is None:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=(
                f"You're not sure how to combine the {item_a.name} and the {item_b.name}."
            ),
        )

    rule, swapped = match
    if swapped:
        item_a, item_b = item_b, item_a  # normalise so A matches input_a

    actor_id = str(action.actor)
    transitions: list[Transition] = []

    # Consume items per rule flags
    if rule.consume_a:
        transitions.append(Transition(
            kind=TransitionKind.ITEM_DESTROYED,
            payload={"object_id": item_a.object_id, "owner_entity": actor_id},
        ))
    if rule.consume_b:
        transitions.append(Transition(
            kind=TransitionKind.ITEM_DESTROYED,
            payload={"object_id": item_b.object_id, "owner_entity": actor_id},
        ))

    # Tag additions to surviving items
    if rule.add_tag_to_a and not rule.consume_a:
        if rule.add_tag_to_a not in item_a.tags:
            transitions.append(Transition(
                kind=TransitionKind.ENTITY_CONDITION_CHANGED,
                payload={
                    "entity_id": actor_id,   # actor owns the item — proxy for tag change
                    "condition": f"item_tag_{item_a.object_id}_{rule.add_tag_to_a}",
                    "ticks": 9999,
                    "_item_tag_add": {"object_id": str(item_a.object_id), "tag": rule.add_tag_to_a},
                },
            ))
    if rule.add_tag_to_b and not rule.consume_b:
        if rule.add_tag_to_b not in item_b.tags:
            # Remove on_fire / frozen if target tag is None (special: douse fire)
            if rule.add_tag_to_b == "null":
                if "on_fire" in item_b.tags:
                    transitions.append(Transition(
                        kind=TransitionKind.ENTITY_CONDITION_CHANGED,
                        payload={
                            "entity_id": actor_id,
                            "condition": f"item_tag_{item_b.object_id}_on_fire",
                            "ticks": 0,
                            "_item_tag_remove": {"object_id": str(item_b.object_id), "tag": "on_fire"},
                        },
                    ))
            else:
                transitions.append(Transition(
                    kind=TransitionKind.ENTITY_CONDITION_CHANGED,
                    payload={
                        "entity_id": actor_id,
                        "condition": f"item_tag_{item_b.object_id}_{rule.add_tag_to_b}",
                        "ticks": 9999,
                        "_item_tag_add": {"object_id": str(item_b.object_id), "tag": rule.add_tag_to_b},
                    },
                ))

    # Attribute updates to surviving items (use a special ITEM_ATTRIBUTE_CHANGED transition)
    if rule.attribute_set_a and not rule.consume_a:
        transitions.append(Transition(
            kind=TransitionKind.ITEM_TRANSFERRED,    # reuse as a no-op carrier
            payload={
                "object_id": item_a.object_id,
                "from_entity": None,
                "to_entity": actor_id,
                "_attribute_patch": rule.attribute_set_a,
            },
        ))
    if rule.attribute_set_b and not rule.consume_b:
        transitions.append(Transition(
            kind=TransitionKind.ITEM_TRANSFERRED,
            payload={
                "object_id": item_b.object_id,
                "from_entity": None,
                "to_entity": actor_id,
                "_attribute_patch": rule.attribute_set_b,
            },
        ))

    # Create result item at actor's feet
    if rule.create_item:
        import uuid
        new_oid = f"craft_{uuid.uuid4().hex[:8]}"
        tmpl = dict(rule.create_item)
        tmpl.setdefault("weight", 0.5)
        tmpl.setdefault("tags", [])
        transitions.append(Transition(
            kind=TransitionKind.ITEM_CREATED,
            payload={
                "object_id": new_oid,
                "name": tmpl.get("name", "unknown item"),
                "tags": tmpl.get("tags", []),
                "weight": tmpl.get("weight", 0.5),
                "bulk": tmpl.get("bulk", 0.5),
                "attributes": tmpl.get("attributes", {}),
                "meta": tmpl.get("meta", {}),
                "position": {"x": actor.position.x, "y": actor.position.y},
                "to_entity": actor_id,  # created directly into actor's inventory
            },
        ))

    # Actor tag
    if rule.add_actor_tag:
        transitions.append(Transition(
            kind=TransitionKind.ENTITY_CONDITION_CHANGED,
            payload={"entity_id": actor_id, "condition": f"tag_{rule.add_actor_tag}",
                     "ticks": 9999, "_tag_add": rule.add_actor_tag},
        ))

    # Actor damage (risky combinations)
    if rule.damage_actor > 0:
        transitions.append(Transition(
            kind=TransitionKind.ENTITY_HEALTH_CHANGED,
            payload={"entity_id": actor_id, "delta": -rule.damage_actor,
                     "cause": "combination_accident", "actor": actor_id},
        ))

    return ValidationResult(valid=True, concrete_transitions=transitions, plausibility=1.0)


# ---------------------------------------------------------------------------
# EXAMINE
# ---------------------------------------------------------------------------

def _compile_examine(action: SemanticAction, world: WorldState) -> ValidationResult:
    """
    EXAMINE kernel — detailed, skill-checked inspection of a visible entity.

    Produces a ``DIALOGUE_SPOKEN`` transition whose payload carries a
    structured ``examine_result`` dict.  The narrator (and the LM narrator
    path) renders it as a rich dossier:

        Name / kind / faction / occupation / emotional state / alertness /
        visible conditions / visible tags / visible equipment / gold fraction /
        visible inventory items / distance / facing direction.

    Skill resolution
    ─────────────────
    perception_score  = actor.attributes["perception"]   (default 50)
    target_stealth    = target.attributes.get("stealth", 0)
    success threshold = 30 + target_stealth // 2

    Below threshold the actor notices only surface facts (name, distance,
    emotional state).  Above threshold they see conditions, tags, gear, and
    faction.

    Range: target must be within sight_range and visible.
    """
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)

    # Resolve target
    if action.target is None:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.MALFORMED_ACTION,
            rejection_detail="EXAMINE requires a target entity or tile.",
        )

    from ..fact_registry import survey_tile
    from ..schemas import Coord

    if isinstance(action.target, Coord):
        result = survey_tile(world, actor, action.target)
        return ValidationResult(
            valid=True,
            concrete_transitions=[
                Transition(
                    kind=TransitionKind.DIALOGUE_SPOKEN,
                    payload={
                        "speaker": action.actor,
                        "content": f"[examine tile {coord_key(action.target)}]",
                        "examine_result": result,
                        "detailed": True,
                    },
                )
            ],
            plausibility=1.0,
        )

    raw_target = str(action.target)
    parts = raw_target.split(",")
    if len(parts) >= 2:
        try:
            tile_coord = Coord(
                x=int(parts[0]),
                y=int(parts[1]),
                z=int(parts[2]) if len(parts) > 2 else 0,
            )
            if grid.is_in_bounds(tile_coord):
                result = survey_tile(world, actor, tile_coord)
                return ValidationResult(
                    valid=True,
                    concrete_transitions=[
                        Transition(
                            kind=TransitionKind.DIALOGUE_SPOKEN,
                            payload={
                                "speaker": action.actor,
                                "content": f"[examine tile {raw_target}]",
                                "examine_result": result,
                                "detailed": True,
                            },
                        )
                    ],
                    plausibility=1.0,
                )
        except ValueError:
            pass

    target = _resolve_entity_target(grid, raw_target)
    if target is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.TARGET_NOT_FOUND)

    if target.position not in visible_from(grid, actor.position, actor.sight_range):
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ENTITY_NOT_VISIBLE)

    dist = actor.position.manhattan(target.position)

    # Skill check
    perception = actor.attributes.get("perception", 50)
    target_stealth = target.attributes.get("stealth", 0)
    threshold = 30 + target_stealth // 2
    detailed = perception >= threshold

    # Build examination result
    result: dict[str, Any] = {
        "name": target.name,
        "kind": target.kind.value,
        "distance": dist,
        "emotional_state": target.emotional_state.value,
        "alertness": target.alertness.value,
        "facing": target.facing.value,
        "alive": target.alive,
    }
    if detailed:
        result.update({
            "occupation": target.occupation,
            "faction": target.faction,
            "conditions": list(target.conditions.keys()),
            "tags": list(target.tags),
            "health_fraction": round(target.health / max(target.max_health, 1), 2),
            "stamina_fraction": round(target.stats.get("stamina", 100) / 100, 2),
            "gold_visible": target.stats.get("gold", 0) > 0,
            "gold_amount": target.stats.get("gold", 0),
            "reputation": target.stats.get("reputation", 0),
            "equipped_weapon": (
                grid.objects[target.equipped_weapon].name
                if target.equipped_weapon and target.equipped_weapon in grid.objects else None
            ),
            "equipped_armor": (
                grid.objects[target.equipped_armor].name
                if target.equipped_armor and target.equipped_armor in grid.objects else None
            ),
            "visible_inventory": [
                grid.objects[oid].name
                for oid in target.inventory
                if oid in grid.objects and "hidden" not in grid.objects[oid].tags
            ][:5],  # cap at 5 items to avoid token bloat
            "social_openness": target.social_openness,
            "role": target.role,
            "drive": target.drive,
        })

    from ..fact_registry import entity_fact_lines

    result["established_facts"] = entity_fact_lines(world, str(target.entity_id))

    return ValidationResult(
        valid=True,
        concrete_transitions=[
            Transition(
                kind=TransitionKind.DIALOGUE_SPOKEN,
                payload={
                    "speaker": action.actor,
                    "listener": str(target.entity_id),
                    "content": f"[examine result for {target.name}]",
                    "examine_result": result,
                    "detailed": detailed,
                },
            )
        ],
        plausibility=1.0,
    )


# ---------------------------------------------------------------------------
# TRADE
# ---------------------------------------------------------------------------

def _compile_trade(action: SemanticAction, world: WorldState) -> ValidationResult:
    """
    TRADE kernel — voluntary economic exchange between two entities.

    Intent encoding (set by the pre-processor or the LM):
      intent.rationale        — items the actor OFFERS (comma-separated names)
      intent.manner           — items the actor REQUESTS from target
      intent.desired_outcome  — ["gold:<N>"] the actor offers in gold,
                                 or ["want:gold:<N>"] the actor wants gold

    The compiler validates asset availability, checks the target's
    willingness (social_openness ≥ 0.3), then emits:
      • ITEM_TRANSFERRED per offered/requested item
      • ENTITY_STAT_CHANGED(stat="gold") for both parties
      • EDGE_UPDATED on the relational graph (trade builds alliance)
    """
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)

    if action.target is None or isinstance(action.target, Coord):
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.MALFORMED_ACTION,
            rejection_detail="TRADE requires an entity target.",
        )
    target = _resolve_entity_target(grid, str(action.target))
    if target is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.TARGET_NOT_FOUND)

    dist = actor.position.manhattan(target.position)
    if dist > 2:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.TARGET_OUT_OF_RANGE,
            rejection_detail=f"Too far away to trade — move closer (distance {dist}).",
        )

    # Willingness check
    # A trade is refused only when the target is unwilling (hostile state OR
    # "closed" social_openness AND poor relationship).
    target_hostile = target.emotional_state.value in ("hostile", "angry")
    target_closed  = target.social_openness in ("closed",)
    actor_rep = actor.stats.get("reputation", 0.0)
    if target_hostile or (target_closed and actor_rep < 0):
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=(
                f"{target.name} isn't interested in trading with you right now."
            ),
        )

    # Parse offer from intent
    offer_gold = 0.0
    want_gold = 0.0
    offer_item_names: list[str] = []
    want_item_names: list[str] = []

    if action.intent:
        raw_offer = (action.intent.rationale or "").strip()
        raw_want = (action.intent.manner or "").strip()
        for chunk in raw_offer.split(","):
            chunk = chunk.strip()
            if chunk.lower().startswith("gold:"):
                try:
                    offer_gold = float(chunk[5:].strip())
                except ValueError:
                    pass
            elif chunk:
                offer_item_names.append(chunk)
        for chunk in raw_want.split(","):
            chunk = chunk.strip()
            if chunk.lower().startswith("gold:"):
                try:
                    want_gold = float(chunk[5:].strip())
                except ValueError:
                    pass
            elif chunk:
                want_item_names.append(chunk)

    from ..economy import infer_gold_from_market, record_trade

    if want_item_names and offer_gold == 0 and not offer_item_names:
        offer_gold = infer_gold_from_market(
            world, want_item_names, region_id=world.active_region_id,
        )
    if offer_item_names and want_gold == 0 and not want_item_names:
        want_gold = infer_gold_from_market(
            world, offer_item_names, region_id=world.active_region_id,
        )

    # Validate actor has offered items + gold
    actor_gold = actor.stats.get("gold", 0.0)
    if offer_gold > actor_gold:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=f"You only have {actor_gold:.0f} gold.",
        )
    actor_id = str(action.actor)
    target_id = str(target.entity_id)

    actor_offer_objs: list[ObjectState] = []
    for name in offer_item_names:
        obj = _find_item_by_name(name, actor, grid)
        if obj is None:
            return ValidationResult(
                valid=False,
                rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
                rejection_detail=f"You don't have '{name}' to offer.",
            )
        actor_offer_objs.append(obj)

    # Validate target has requested items + gold
    target_gold = target.stats.get("gold", 0.0)
    if want_gold > target_gold:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=f"{target.name} doesn't have {want_gold:.0f} gold.",
        )
    target_offer_objs: list[ObjectState] = []
    for name in want_item_names:
        obj = _find_item_by_name(name, target, grid)
        if obj is None:
            return ValidationResult(
                valid=False,
                rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
                rejection_detail=f"{target.name} doesn't have '{name}'.",
            )
        target_offer_objs.append(obj)

    transitions: list[Transition] = []

    # Transfer items actor → target
    for obj in actor_offer_objs:
        transitions.append(Transition(
            kind=TransitionKind.ITEM_TRANSFERRED,
            payload={"object_id": obj.object_id, "from_entity": actor_id, "to_entity": target_id},
        ))

    # Transfer items target → actor
    for obj in target_offer_objs:
        transitions.append(Transition(
            kind=TransitionKind.ITEM_TRANSFERRED,
            payload={"object_id": obj.object_id, "from_entity": target_id, "to_entity": actor_id},
        ))

    # Gold transfers
    if offer_gold > 0:
        transitions.append(Transition(
            kind=TransitionKind.ENTITY_STAT_CHANGED,
            payload={"entity_id": actor_id, "stat": "gold", "delta": -offer_gold,
                     "cause": "trade_payment", "actor": actor_id},
        ))
        transitions.append(Transition(
            kind=TransitionKind.ENTITY_STAT_CHANGED,
            payload={"entity_id": target_id, "stat": "gold", "delta": offer_gold,
                     "cause": "trade_payment", "actor": actor_id},
        ))
    if want_gold > 0:
        transitions.append(Transition(
            kind=TransitionKind.ENTITY_STAT_CHANGED,
            payload={"entity_id": target_id, "stat": "gold", "delta": -want_gold,
                     "cause": "trade_payment", "actor": actor_id},
        ))
        transitions.append(Transition(
            kind=TransitionKind.ENTITY_STAT_CHANGED,
            payload={"entity_id": actor_id, "stat": "gold", "delta": want_gold,
                     "cause": "trade_payment", "actor": actor_id},
        ))

    # Reputation boost — trade is prosocial
    transitions.append(Transition(
        kind=TransitionKind.ENTITY_STAT_CHANGED,
        payload={"entity_id": actor_id, "stat": "reputation", "delta": 2.0,
                 "cause": "completed_trade", "actor": actor_id},
    ))

    # Strengthen the relational edge
    transitions.append(Transition(
        kind=TransitionKind.EDGE_UPDATED,
        payload={
            "source": actor_id,
            "target": target_id,
            "edge_kind": EdgeKind.ALLY_OF.value,
            "delta": 0.15,
        },
    ))

    for name in want_item_names + offer_item_names:
        record_trade(
            world,
            name.lower().replace(" ", "_"),
            region_id=world.active_region_id,
        )

    return ValidationResult(valid=True, concrete_transitions=transitions, plausibility=1.0)


# ---------------------------------------------------------------------------
# STEAL
# ---------------------------------------------------------------------------

def _compile_steal(action: SemanticAction, world: WorldState) -> ValidationResult:
    """
    STEAL kernel — covert item acquisition from another entity.

    Success conditions:
      1. Target must not have the actor in their vision cone (facing check).
      2. Stealth check: actor.stealth vs (target.perception + target.alertness_bonus).

    Outcomes:
      SUCCESS → ITEM_TRANSFERRED + low NOISE_EVENT
      FAILURE → ENTITY_ALERTNESS_CHANGED (target notices) +
                EDGE_UPDATED (relationship damaged) +
                ENTITY_STAT_CHANGED(reputation, -10)
    """
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)

    if action.target is None or isinstance(action.target, Coord):
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.MALFORMED_ACTION,
            rejection_detail="STEAL requires an entity target.",
        )
    target = _resolve_entity_target(grid, str(action.target))
    if target is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.TARGET_NOT_FOUND)

    dist = actor.position.manhattan(target.position)
    if dist > 1:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.TARGET_OUT_OF_RANGE,
            rejection_detail="You must be adjacent to steal from someone.",
        )

    # Find the item to steal
    item_hint = (action.intent.rationale or "") if action.intent else ""
    stolen: Optional[ObjectState] = None
    if item_hint:
        stolen = _find_item_by_name(item_hint, target, grid)
    if stolen is None and target.inventory:
        stolen = grid.objects.get(target.inventory[0])
    if stolen is None:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=f"{target.name} has nothing to steal.",
        )

    # Check if actor is in target's vision cone
    from ..spatial import _in_facing_cone, _cone_cos
    dx = actor.position.x - target.position.x
    dy = actor.position.y - target.position.y
    target_half_cos = _cone_cos(target.fov_degrees)
    actor_visible_to_target = _in_facing_cone(dx, dy, target.facing, target_half_cos)

    # Alertness bonus
    alertness_bonus = {
        "unaware": -20, "low": -10, "medium": 0, "high": 15, "combat": 30,
    }.get(target.alertness.value, 0)

    actor_stealth = actor.attributes.get("stealth", 50)
    target_perception = target.attributes.get("perception", 50) + alertness_bonus

    # Guaranteed fail if actor is in vision cone
    seed = f"{action.action_id}_{world.tick}_steal"
    roll = _seeded_float(seed) * 100
    success = not actor_visible_to_target and (actor_stealth + roll * 0.3 > target_perception)

    actor_id = str(action.actor)
    target_id = str(target.entity_id)

    if success:
        return ValidationResult(
            valid=True,
            concrete_transitions=[
                Transition(
                    kind=TransitionKind.ITEM_TRANSFERRED,
                    payload={"object_id": stolen.object_id,
                             "from_entity": target_id, "to_entity": actor_id},
                ),
                Transition(
                    kind=TransitionKind.NOISE_EVENT,
                    payload={"source_id": actor_id,
                             "source_pos": {"x": actor.position.x, "y": actor.position.y},
                             "noise_level": 10},
                ),
            ],
            plausibility=1.0,
        )
    else:
        return ValidationResult(
            valid=True,
            concrete_transitions=[
                Transition(
                    kind=TransitionKind.ENTITY_ALERTNESS_CHANGED,
                    payload={"entity_id": target_id,
                             "from": target.alertness.value,
                             "to": AlertnessLevel.HIGH.value},
                ),
                Transition(
                    kind=TransitionKind.EDGE_UPDATED,
                    payload={
                        "source": target_id,
                        "target": actor_id,
                        "edge_kind": EdgeKind.WARY_OF.value,
                        "delta": 0.3,
                    },
                ),
                Transition(
                    kind=TransitionKind.ENTITY_STAT_CHANGED,
                    payload={"entity_id": actor_id, "stat": "reputation", "delta": -10.0,
                             "cause": "caught_stealing", "actor": actor_id},
                ),
            ],
            plausibility=1.0,
            # Mark as "valid but failed" — narrator uses this to distinguish outcomes
            rejection_detail="__steal_failed__",
        )


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------

def _compile_rest(action: SemanticAction, world: WorldState) -> ValidationResult:
    """
    REST kernel — the actor takes a moment to recover stamina and health.

    Valid only when:
      • alertness < HIGH (not in combat or actively threatened)
      • actor is alive

    Recovery per rest action:
      stamina: +20 (capped at 100)
      health:  +5 (capped at max_health)
      Adds the "resting" condition (3 ticks, cleared on next non-wait action).
    """
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)

    _active_alertness = {AlertnessLevel.HIGH, AlertnessLevel.COMBAT}
    if actor.alertness in _active_alertness:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail="You can't rest while in danger.",
        )

    actor_id = str(action.actor)
    current_stamina = actor.stats.get("stamina", 100.0)
    stamina_recovery = min(20.0, 100.0 - current_stamina)
    health_recovery = min(5, actor.max_health - actor.health)

    transitions: list[Transition] = []

    if stamina_recovery > 0:
        transitions.append(Transition(
            kind=TransitionKind.ENTITY_STAT_CHANGED,
            payload={"entity_id": actor_id, "stat": "stamina",
                     "delta": stamina_recovery, "cause": "rest", "actor": actor_id},
        ))

    if health_recovery > 0:
        transitions.append(Transition(
            kind=TransitionKind.ENTITY_HEALTH_CHANGED,
            payload={"entity_id": actor_id, "delta": health_recovery,
                     "cause": "rest", "actor": actor_id},
        ))

    transitions.append(Transition(
        kind=TransitionKind.ENTITY_CONDITION_CHANGED,
        payload={"entity_id": actor_id, "condition": "resting", "ticks": 3},
    ))

    return ValidationResult(valid=True, concrete_transitions=transitions, plausibility=1.0)


# ---------------------------------------------------------------------------
# CRAFT  (open-world object synthesis)
# ---------------------------------------------------------------------------

def _compile_craft(action: SemanticAction, world: WorldState) -> ValidationResult:
    """
    CRAFT kernel — player synthesises a new object from materials they hold.

    This is the gateway to open-world creativity: an item that was never
    authored in objects.yaml can enter the canonical world state if the
    engine validates it as physically plausible.

    Validation pipeline
    ──────────────────
    1. The player describes what they want to make (intent.rationale) and
       optionally which items to use (intent.manner, comma-separated names).
    2. The engine checks whether the actor's inventory covers the input
       materials and verifies against ``world.config.synthesis_hints``:
         • At least one hint's ``required_input_tags`` must be covered by
           the union of tags from the consumed items.
         • The proposed output's tags must intersect the hint's
           ``output_tags``.
    3. If no LM-proposed ``synthesized_object`` is on the action, the
       engine falls back to a simple rule-based output derived from the
       inputs (name = "crude <rationale>", attributes derived from best
       matching hint).
    4. Attribute values are capped at ``hint.max_attribute`` and weight/
       bulk are bounded.  Any attribute above cap is silently clamped.
    5. On success:
       • Consumed input items → ITEM_DESTROYED per item.
       • New object → ITEM_SYNTHESIZED (writes a real ObjectState into
         the spatial grid and actor's inventory).
       • Stamina cost → ENTITY_STAT_CHANGED(stat="stamina").

    The LM adapter (when active) populates ``action.proposed_effects`` with
    an ITEM_SYNTHESIZED proposal carrying ``synthesized_object``; when no
    LM is present the rule-based path fires instead.
    """
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)

    actor_id = str(action.actor)

    # Parse intent
    output_name = (action.intent.rationale or "").strip() if action.intent else ""
    input_hint_raw = (action.intent.manner or "").strip() if action.intent else ""

    if not output_name:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.MALFORMED_ACTION,
            rejection_detail="Specify what you want to craft (e.g. 'craft a bandage from cloth').",
        )

    # Resolve input items actor wants to use
    input_item_names: list[str] = [s.strip() for s in input_hint_raw.split(",") if s.strip()]
    consumed_objs: list[ObjectState] = []
    for name in input_item_names:
        obj = _find_item_by_name(name, actor, grid)
        if obj is None:
            return ValidationResult(
                valid=False,
                rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
                rejection_detail=f"You don't have '{name}' in your inventory.",
            )
        consumed_objs.append(obj)

    # If actor gave no specific inputs, auto-select any items with
    # "raw_material", "material", or "component" tags as candidates
    if not consumed_objs:
        for oid in actor.inventory:
            obj = grid.objects.get(oid)
            if obj and any(t in obj.tags for t in ("raw_material", "material", "component", "scrap")):
                consumed_objs.append(obj)
        if not consumed_objs:
            return ValidationResult(
                valid=False,
                rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
                rejection_detail="You have no suitable materials to craft with.",
            )

    # Check for a synthesis hint that covers the input materials
    input_tag_union: set[str] = set()
    for obj in consumed_objs:
        input_tag_union.update(obj.tags)

    # Infer desired output tags from the name (simple heuristic)
    name_lower = output_name.lower()
    desired_tags: set[str] = set()
    _TAG_KEYWORDS = {
        "bandage": ["bandage", "consumable"],
        "torch": ["light_source", "flammable", "wooden"],
        "knife": ["weapon", "sharp", "throwable"],
        "shiv": ["weapon", "sharp", "throwable"],
        "club": ["weapon", "wooden"],
        "rope": ["rope", "tool"],
        "trap": ["trap", "tool"],
        "tool": ["tool"],
        "armor": ["armor"],
        "shield": ["armor"],
        "bow": ["weapon", "ranged", "wooden"],
        "arrow": ["projectile", "throwable", "wooden"],
        "potion": ["consumable", "liquid", "potion"],
        "salve": ["consumable", "healing"],
        "candle": ["light_source", "wax", "flammable"],
        "net": ["trap", "tool"],
        "hook": ["tool", "metal"],
        "key": ["tool", "metal"],
    }
    for kw, tags in _TAG_KEYWORDS.items():
        if kw in name_lower:
            desired_tags.update(tags)
    if not desired_tags:
        desired_tags = {"tool"}  # default assumption

    # Find best matching synthesis hint
    cfg = world.config
    best_hint = None
    for hint in (cfg.synthesis_hints if cfg else []):
        req = set(hint.required_input_tags)
        out = set(hint.output_tags)
        if req <= input_tag_union and desired_tags & out:
            best_hint = hint
            break

    # If no hint, allow if player consumed something and asked for
    # something plausible-sounding (graceful permissiveness).
    # The output is tagged "improvised" and gets weak attributes.
    if best_hint is None and not consumed_objs:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=(
                f"You can't figure out how to make a '{output_name}' "
                f"from what you're holding."
            ),
        )

    # Build the proposed object dict — prefer LM proposal if present
    proposed_schema: Optional[dict[str, Any]] = None
    for eff in (action.proposed_effects or []):
        if eff.kind == TransitionKind.ITEM_SYNTHESIZED.value and eff.synthesized_object:
            proposed_schema = eff.synthesized_object
            break

    if proposed_schema is None:
        # Rule-based fallback: derive from inputs
        weight = round(sum(o.weight for o in consumed_objs) * 0.5, 2)
        bulk = round(sum(o.bulk for o in consumed_objs) * 0.5, 2)
        out_tags = list(desired_tags) + ["crafted"]
        if best_hint is None:
            out_tags.append("improvised")
        # Derive a base attribute from the best consumed item's damage/value
        base_attr = max(
            (o.attributes.get("damage", o.attributes.get("value", 5))
             for o in consumed_objs),
            default=5
        )
        attr_cap = best_hint.max_attribute if best_hint else 25
        attrs: dict[str, int] = {}
        if "weapon" in desired_tags:
            attrs["damage"] = min(int(base_attr * 0.7), attr_cap)
        if "consumable" in desired_tags or "bandage" in out_tags:
            attrs["heal"] = min(max(int(base_attr * 0.3), 5), attr_cap)
        if "armor" in desired_tags:
            attrs["armor"] = min(int(base_attr * 0.4), attr_cap)
        proposed_schema = {
            "name": output_name,
            "tags": out_tags,
            "weight": min(weight, best_hint.max_weight if best_hint else 10.0),
            "bulk": min(bulk, best_hint.max_bulk if best_hint else 10.0),
            "attributes": attrs,
            "meta": {"description": f"A crude {output_name} crafted on the spot."},
        }
    else:
        # Sanitise LM proposal — enforce attribute caps
        attr_cap = best_hint.max_attribute if best_hint else 30
        for k in list(proposed_schema.get("attributes", {}).keys()):
            v = proposed_schema["attributes"][k]
            if isinstance(v, (int, float)):
                proposed_schema["attributes"][k] = min(int(v), attr_cap)
        w_cap = best_hint.max_weight if best_hint else 15.0
        b_cap = best_hint.max_bulk if best_hint else 15.0
        proposed_schema["weight"] = min(float(proposed_schema.get("weight", 1.0)), w_cap)
        proposed_schema["bulk"] = min(float(proposed_schema.get("bulk", 1.0)), b_cap)

    # Assign a real object id
    import uuid as _uuid
    new_oid = f"obj_{_uuid.uuid4().hex[:8]}"
    proposed_schema["object_id"] = new_oid

    stamina_cost = best_hint.stamina_cost if best_hint else 15.0

    transitions: list[Transition] = []

    # Consume input items
    for obj in consumed_objs:
        transitions.append(Transition(
            kind=TransitionKind.ITEM_DESTROYED,
            payload={"object_id": obj.object_id, "cause": "crafted"},
        ))

    # Stamina cost
    transitions.append(Transition(
        kind=TransitionKind.ENTITY_STAT_CHANGED,
        payload={"entity_id": actor_id, "stat": "stamina",
                 "delta": -stamina_cost, "cause": "crafting", "actor": actor_id},
    ))

    # Create the new item
    transitions.append(Transition(
        kind=TransitionKind.ITEM_SYNTHESIZED,
        payload={
            **proposed_schema,
            "owner_entity_id": actor_id,
            "cause": "craft_action",
        },
    ))

    return ValidationResult(valid=True, concrete_transitions=transitions, plausibility=1.0)


# ---------------------------------------------------------------------------
# MARK  (write a persistent mark onto a tile)
# ---------------------------------------------------------------------------

def _compile_mark(action: SemanticAction, world: WorldState) -> ValidationResult:
    """
    MARK kernel — scratch, draw, or write something on the tile the actor
    is standing on or an adjacent tile.

    What gets marked:
      intent.rationale — the mark text (e.g. "X", "here be danger", "my rune")
      intent.manner    — optional target coord hint ("the wall to the north")

    The mark is stored in ``tile.marks`` and persists across ticks.
    It is visible to any entity that EXAMINEs the tile (shown in narration).
    Multiple marks can coexist on the same tile.

    Removing a mark: intent.desired_outcome contains "remove" or "erase".
    """
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)

    mark_text = (action.intent.rationale or "").strip() if action.intent else ""
    if not mark_text:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.MALFORMED_ACTION,
            rejection_detail="Specify what to mark (e.g. 'mark an X on the floor').",
        )

    # Resolve target coordinate
    target_coord = actor.position  # default: actor's tile
    if isinstance(action.target, Coord):
        target_coord = action.target
    elif action.intent and action.intent.manner:
        # Parse simple directional hint
        manner = action.intent.manner.lower()
        dx, dy = 0, 0
        if "north" in manner:
            dy = -1
        elif "south" in manner:
            dy = 1
        if "east" in manner or "right" in manner:
            dx = 1
        elif "west" in manner or "left" in manner:
            dx = -1
        if dx or dy:
            target_coord = Coord(x=actor.position.x + dx, y=actor.position.y + dy)

    if not grid.is_in_bounds(target_coord):
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.TARGET_OUT_OF_RANGE,
            rejection_detail="That tile is out of bounds.",
        )
    if actor.position.manhattan(target_coord) > 1:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.TARGET_OUT_OF_RANGE,
            rejection_detail="You can only mark tiles adjacent to you.",
        )

    removing = action.intent and any(
        w in (action.intent.desired_outcome or []) for w in ("remove", "erase", "clear")
    )

    return ValidationResult(
        valid=True,
        concrete_transitions=[
            Transition(
                kind=TransitionKind.TILE_MARKED,
                payload={
                    "x": target_coord.x,
                    "y": target_coord.y,
                    "mark": mark_text[:80],
                    "author_entity_id": str(action.actor),
                    "remove": removing,
                },
            )
        ],
        plausibility=1.0,
    )


# ---------------------------------------------------------------------------
# BARRICADE  (place an impassable structure at a tile)
# ---------------------------------------------------------------------------

def _compile_barricade(action: SemanticAction, world: WorldState) -> ValidationResult:
    """
    BARRICADE kernel — use items from inventory to create a physical
    structure that blocks passage.

    Validation:
      • Actor must be on or adjacent to the target tile.
      • Target tile must currently be passable floor (can't double-barricade).
      • Actor must have at least one "raw_material" or "wooden" item, OR
        any item with bulk >= 1.0 (big enough to block a path).

    The consumed item (heaviest/bulkiest in inventory that isn't equipped)
    is destroyed and becomes the barricade structure.

    The barricade persists as an impassable ObjectState until destroyed or
    moved (via ATTACK, TAKE with sufficient strength, etc.).
    """
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)

    # Resolve target tile
    target_coord = actor.position
    if isinstance(action.target, Coord):
        target_coord = action.target
    elif action.intent and action.intent.manner:
        manner = action.intent.manner.lower()
        dx, dy = 0, 0
        if "north" in manner:
            dy = -1
        elif "south" in manner:
            dy = 1
        if "east" in manner or "right" in manner:
            dx = 1
        elif "west" in manner or "left" in manner:
            dx = -1
        if dx or dy:
            target_coord = Coord(x=actor.position.x + dx, y=actor.position.y + dy)

    if not grid.is_in_bounds(target_coord):
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.TARGET_OUT_OF_RANGE,
            rejection_detail="That tile is out of bounds.",
        )
    if actor.position.manhattan(target_coord) > 1:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.TARGET_OUT_OF_RANGE,
            rejection_detail="You can only barricade adjacent tiles.",
        )

    # Check target passable
    tile = grid.tile_at(target_coord)
    if not tile.passable:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail="That tile is already blocked.",
        )
    # Check no existing impassable structure there
    for obj in grid.objects.values():
        if obj.position == target_coord and not obj.passable:
            return ValidationResult(
                valid=False,
                rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
                rejection_detail="Something is already blocking that tile.",
            )

    # Find something to use as the barricade material
    # Prefer bulky items, then raw_materials, then any non-equipped item
    best_item: Optional[ObjectState] = None
    for oid in actor.inventory:
        if oid == actor.equipped_weapon or oid == actor.equipped_armor:
            continue
        obj = grid.objects.get(oid)
        if obj is None:
            continue
        if best_item is None or obj.bulk > best_item.bulk:
            best_item = obj

    if best_item is None:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail="You have nothing to use as a barricade.",
        )

    import uuid as _uuid
    barricade_oid = f"obj_{_uuid.uuid4().hex[:8]}"
    barricade_name = f"barricade of {best_item.name}"

    stamina_cost = max(10.0, best_item.bulk * 3)

    return ValidationResult(
        valid=True,
        concrete_transitions=[
            Transition(
                kind=TransitionKind.ITEM_DESTROYED,
                payload={"object_id": best_item.object_id, "cause": "used_as_barricade"},
            ),
            Transition(
                kind=TransitionKind.ENTITY_STAT_CHANGED,
                payload={"entity_id": str(action.actor), "stat": "stamina",
                         "delta": -stamina_cost, "cause": "barricade", "actor": str(action.actor)},
            ),
            Transition(
                kind=TransitionKind.STRUCTURE_CREATED,
                payload={
                    "x": target_coord.x,
                    "y": target_coord.y,
                    "object_id": barricade_oid,
                    "name": barricade_name,
                    "tags": ["structure", "barricade", "destructible"],
                    "passable": False,
                    "transparent": False,
                    "weight": best_item.weight,
                    "bulk": best_item.bulk,
                    "attributes": {"durability": 30},
                    "meta": {"material": best_item.meta.get("material", "mixed"),
                             "description": f"A crude barricade made from {best_item.name}."},
                    "author_entity_id": str(action.actor),
                },
            ),
        ],
        plausibility=1.0,
    )


# ---------------------------------------------------------------------------
# UNLOCK / LOCK  (change locked state of a structure or door tile)
# ---------------------------------------------------------------------------

def _compile_unlock(action: SemanticAction, world: WorldState) -> ValidationResult:
    """
    UNLOCK / LOCK kernel — change the locked/passable state of a door tile
    or a structure object.

    Mechanisms:
      • Actor has an item tagged "key" in inventory → automatic success.
      • No key → skill check: actor.attributes["perception"] or "stealth"
        vs difficulty 50 (picking a lock).

    On success: emits STRUCTURE_MODIFIED(set_passable=True/False) +
    ENVIRONMENT_STATE_CHANGED(key="locked", value=False/True) on the tile.
    """
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)

    # Determine locking vs unlocking from verb
    locking = action.verb.lower() in ("lock", "seal", "bolt")

    # Resolve target
    target_coord = None
    target_obj: Optional[ObjectState] = None
    if isinstance(action.target, Coord):
        target_coord = action.target
    elif isinstance(action.target, str):
        target_obj = grid.objects.get(ObjectId(action.target))
        if target_obj and target_obj.position:
            target_coord = target_obj.position

    # Default: the tile immediately in front of the actor
    if target_coord is None:
        facing_vec = {
            FacingDirection.NORTH: (0, -1), FacingDirection.SOUTH: (0, 1),
            FacingDirection.EAST: (1, 0), FacingDirection.WEST: (-1, 0),
            FacingDirection.NE: (1, -1), FacingDirection.NW: (-1, -1),
            FacingDirection.SE: (1, 1), FacingDirection.SW: (-1, 1),
        }.get(actor.facing, (0, 1))
        target_coord = Coord(
            x=actor.position.x + facing_vec[0],
            y=actor.position.y + facing_vec[1],
        )

    if not grid.is_in_bounds(target_coord):
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.TARGET_OUT_OF_RANGE,
            rejection_detail="Nothing to unlock there.",
        )

    dist = actor.position.manhattan(target_coord)
    if dist > 1:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.TARGET_OUT_OF_RANGE,
            rejection_detail="You must be adjacent to interact with a lock.",
        )

    tile = grid.tile_at(target_coord)
    current_locked = tile.env.get("locked", not tile.passable)

    if locking and not current_locked:
        pass  # we'll lock it
    elif not locking and not current_locked:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail="It's already unlocked.",
        )

    # Has a key?
    has_key = any(
        "key" in (grid.objects.get(oid) or type("", (), {"tags": []})()).tags
        for oid in actor.inventory
    )

    actor_id = str(action.actor)
    if not has_key:
        # Lockpick attempt
        skill = actor.attributes.get("stealth", 30)  # stealth proxies dexterity
        seed = f"{action.action_id}_{world.tick}_lock"
        roll = _seeded_float(seed) * 100
        if skill + roll * 0.3 < 50:
            return ValidationResult(
                valid=False,
                rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
                rejection_detail="You fumble with the lock — it doesn't yield.",
            )

    new_passable = locking is False  # unlock → passable=True; lock → passable=False
    transitions: list[Transition] = [
        Transition(
            kind=TransitionKind.ENVIRONMENT_STATE_CHANGED,
            payload={"x": target_coord.x, "y": target_coord.y,
                     "key": "locked", "value": locking, "cause": action.verb,
                     "author": actor_id},
        ),
        Transition(
            kind=TransitionKind.STRUCTURE_MODIFIED,
            payload={"x": target_coord.x, "y": target_coord.y,
                     "set_passable": new_passable,
                     "add_tags": ["unlocked"] if not locking else ["locked"],
                     "remove_tags": ["locked"] if not locking else ["unlocked"],
                     "cause": action.verb},
        ),
    ]
    return ValidationResult(valid=True, concrete_transitions=transitions, plausibility=1.0)


# ---------------------------------------------------------------------------
# IGNITE  (light something in place — campfire, torch, candle, brazier)
# ---------------------------------------------------------------------------

def _compile_ignite(action: SemanticAction, world: WorldState) -> ValidationResult:
    """
    IGNITE kernel — light a flammable structure or loose object in place.

    Valid when:
      • Actor or nearby tile has a fire source (lit torch, fire tag, or
        item tagged "fire_source" in inventory).
      • Target object / tile has a "flammable" tag.

    Effects:
      • ENVIRONMENT_STATE_CHANGED(key="lit", value=True) on the tile.
      • STRUCTURE_MODIFIED(add_tags=["on_fire", "light_source"]) on the obj.
      • ENTITY_CONDITION_CHANGED("illuminated", +5 ticks) to surrounding entities.
    """
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)

    # Find target object (by name hint or action.target)
    target_obj: Optional[ObjectState] = None
    if isinstance(action.target, str):
        target_obj = grid.objects.get(ObjectId(action.target))
    if target_obj is None and action.intent and action.intent.rationale:
        target_obj = _find_item_by_name(action.intent.rationale, actor, grid)
    # Also check ground objects adjacent to actor
    if target_obj is None:
        for obj in grid.objects.values():
            if obj.position and "flammable" in obj.tags:
                dist = actor.position.manhattan(obj.position)
                if dist <= 1:
                    target_obj = obj
                    break

    if target_obj is None:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.TARGET_NOT_FOUND,
            rejection_detail="Nothing flammable nearby to ignite.",
        )

    if "flammable" not in target_obj.tags and "wooden" not in target_obj.tags:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=f"The {target_obj.name} doesn't seem flammable.",
        )

    # Check actor has a fire source
    has_fire_source = any(
        any(t in (grid.objects.get(oid) or type("", (), {"tags": []})()).tags
            for t in ("on_fire", "fire_source", "light_source"))
        for oid in actor.inventory
    )
    # Also check if actor is adjacent to an on_fire object
    if not has_fire_source:
        for obj in grid.objects.values():
            if obj.position and "on_fire" in obj.tags:
                if actor.position.manhattan(obj.position) <= 1:
                    has_fire_source = True
                    break

    if not has_fire_source:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail="You have no fire source to light it with.",
        )

    transitions: list[Transition] = [
        Transition(
            kind=TransitionKind.STRUCTURE_MODIFIED,
            payload={
                "object_id": target_obj.object_id,
                "add_tags": ["on_fire", "light_source"],
                "remove_tags": [],
                "attribute_set": {"temperature": 600, "light_radius": 3},
                "cause": "ignited",
            },
        ),
    ]
    if target_obj.position:
        transitions.append(Transition(
            kind=TransitionKind.ENVIRONMENT_STATE_CHANGED,
            payload={
                "x": target_obj.position.x, "y": target_obj.position.y,
                "key": "lit", "value": True, "cause": "ignited",
                "author": str(action.actor),
            },
        ))

    return ValidationResult(valid=True, concrete_transitions=transitions, plausibility=1.0)


def _compile_extinguish(
    action: SemanticAction, world: WorldState
) -> ValidationResult:
    """EXTINGUISH / quench — remove on_fire from a nearby object."""
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)

    target_obj: Optional[ObjectState] = None
    if isinstance(action.target, str):
        target_obj = grid.objects.get(ObjectId(action.target))
    if target_obj is None:
        for obj in grid.objects.values():
            if obj.position and "on_fire" in obj.tags:
                if actor.position.manhattan(obj.position) <= 2:
                    target_obj = obj
                    break

    if target_obj is None:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.TARGET_NOT_FOUND,
            rejection_detail="Nothing is burning nearby.",
        )

    if "on_fire" not in target_obj.tags:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=f"The {target_obj.name} is not on fire.",
        )

    transitions: list[Transition] = [
        Transition(
            kind=TransitionKind.STRUCTURE_MODIFIED,
            payload={
                "object_id": target_obj.object_id,
                "add_tags": [],
                "remove_tags": ["on_fire", "light_source"],
                "attribute_set": {"temperature": 40},
                "cause": "extinguished",
            },
        ),
    ]
    if target_obj.position:
        transitions.append(Transition(
            kind=TransitionKind.ENVIRONMENT_STATE_CHANGED,
            payload={
                "x": target_obj.position.x,
                "y": target_obj.position.y,
                "key": "lit",
                "value": False,
                "cause": "extinguish",
                "author": str(action.actor),
            },
        ))
    return ValidationResult(valid=True, concrete_transitions=transitions, plausibility=1.0)


def _compile_give(
    action: SemanticAction,
    world: WorldState,
) -> ValidationResult:
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(
            valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND
        )
    if action.target is None:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.MALFORMED_ACTION,
            rejection_detail="GIVE requires a target.",
        )
    # Target is the object id encoded in intent, target field is the recipient
    recipient_id = EntityId(str(action.target))
    recipient = grid.entities.get(recipient_id)
    if recipient is None:
        return ValidationResult(
            valid=False, rejection_reason=RejectionReason.TARGET_NOT_FOUND
        )
    object_id_str = action.intent.desired_outcome[0] if action.intent.desired_outcome else None
    if object_id_str is None or ObjectId(object_id_str) not in actor.inventory:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail="Actor does not have the specified item.",
        )
    object_id = ObjectId(object_id_str)
    return ValidationResult(
        valid=True,
        concrete_transitions=[
            Transition(
                kind=TransitionKind.ITEM_TRANSFERRED,
                payload={
                    "object_id": object_id,
                    "from_entity": action.actor,
                    "to_entity": recipient_id,
                },
            )
        ],
        plausibility=1.0,
    )


def _classify_consent(
    actor: EntityState,
    target: EntityState,
    world: WorldState,
    style_aggression: int,
) -> ConsentState:
    """
    Derive a ConsentState from actor↔target relational edges and the
    action's aggression. Pure function over canonical state — no English
    keywords are read. Pack data determines outcomes by populating
    INTIMATE / WARY_OF / ENEMY_OF / HOSTILE edges.

    Rules (in priority order):
      1. Aggression ≥ 70 and no mutual INTIMATE      → HOSTILE
      2. ENEMY_OF / FEARS / WARY_OF from target→actor → REFUSED
      3. Mutual INTIMATE                             → WELCOMED
      4. target.social_openness == "welcoming"
         and aggression < 30                         → WELCOMED
      5. target.social_openness == "closed"          → REFUSED
      6. otherwise                                   → UNWELCOMED
    """
    from ..relational import get_edges

    a_to_t = {e.kind for e in get_edges(world.relational, actor.entity_id, target.entity_id)}
    t_to_a = {e.kind for e in get_edges(world.relational, target.entity_id, actor.entity_id)}

    mutual_intimate = EdgeKind.INTIMATE in a_to_t and EdgeKind.INTIMATE in t_to_a

    if style_aggression >= 70 and not mutual_intimate:
        return ConsentState.HOSTILE

    refusing_kinds = {EdgeKind.ENEMY_OF, EdgeKind.FEARS, EdgeKind.WARY_OF}
    if t_to_a & refusing_kinds:
        return ConsentState.REFUSED

    if mutual_intimate:
        return ConsentState.WELCOMED

    openness = (target.social_openness or "guarded").lower()
    if openness == "welcoming" and style_aggression < 30:
        return ConsentState.WELCOMED
    if openness == "closed":
        return ConsentState.REFUSED

    return ConsentState.UNWELCOMED


def _compile_contact(
    action: SemanticAction,
    world: WorldState,
) -> ValidationResult:
    """
    Resolve a non-combat physical interaction.

    The compiler does NOT decide how the target reacts — that is the
    target's own turn. We record a CONTACT_INITIATED transition and
    update the target's alertness/emotional state to reflect their
    momentary disposition. The target's policy or the player will then
    choose accept / refuse / retreat / retaliate on the next turn.

    Physical proximity is required (adjacent tiles). High-aggression
    contact additionally degrades health slightly so that hostile
    grappling has some material weight, but the engine does not
    fabricate worse outcomes than that — the rest is up to the target.
    """
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(
            valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND
        )
    if action.target is None or isinstance(action.target, Coord):
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.MALFORMED_ACTION,
            rejection_detail="CONTACT requires an EntityId target.",
        )
    target = _resolve_entity_target(grid, str(action.target))
    if target is None:
        return ValidationResult(
            valid=False, rejection_reason=RejectionReason.TARGET_NOT_FOUND
        )
    target_id = target.entity_id
    if target.position not in visible_from(grid, actor.position, actor.sight_range):
        return ValidationResult(
            valid=False, rejection_reason=RejectionReason.ENTITY_NOT_VISIBLE
        )
    dist = actor.position.manhattan(target.position)
    if dist > 1:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.TARGET_OUT_OF_RANGE,
            rejection_detail=(
                f"Contact requires adjacency; target is {dist} tiles away."
            ),
        )

    aggression = action.style.aggression if action.style else 30
    consent = _classify_consent(actor, target, world, aggression)
    surprise = target.alertness in (AlertnessLevel.UNAWARE, AlertnessLevel.LOW)

    # Always record the contact event itself.
    transitions: list[Transition] = [
        Transition(
            kind=TransitionKind.CONTACT_INITIATED,
            payload={
                "actor": action.actor,
                "target": target_id,
                "manner": (action.intent.manner if action.intent else None) or "",
                "consent_state": consent.value,
                "surprise": surprise,
                "aggression": aggression,
            },
        )
    ]

    # Adjust the target's momentary disposition.
    new_emotional = _emotional_for_consent(consent, target.emotional_state)
    if new_emotional != target.emotional_state:
        transitions.append(
            Transition(
                kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED,
                payload={
                    "entity_id": target_id,
                    "from": target.emotional_state.value,
                    "to": new_emotional.value,
                },
            )
        )

    new_alertness = _alertness_for_consent(consent, target.alertness)
    if new_alertness != target.alertness:
        transitions.append(
            Transition(
                kind=TransitionKind.ENTITY_ALERTNESS_CHANGED,
                payload={
                    "entity_id": target_id,
                    "from": target.alertness.value,
                    "to": new_alertness.value,
                },
            )
        )

    # Only HOSTILE contact directly inflicts (small) physical harm. Even
    # then it is intentionally light — the bulk of the consequence comes
    # from the target's own next action (retaliate, flee, etc.).
    if consent == ConsentState.HOSTILE:
        seed = f"{action.action_id}_{world.tick}_grapple"
        scuff = 1 + int(_seeded_float(seed) * 4)
        transitions.append(
            Transition(
                kind=TransitionKind.ENTITY_HEALTH_CHANGED,
                payload={
                    "entity_id": target_id,
                    "delta": -scuff,
                    "cause": "unwanted_contact",
                    "actor": action.actor,
                },
            )
        )

    return ValidationResult(
        valid=True,
        concrete_transitions=transitions,
        plausibility=1.0,
    )


# ---------------------------------------------------------------------------
# Contact verb classification
#
# Maps freeform verb strings to the aggression level that should be passed
# to _compile_contact.  Aggression is the primary input to _classify_consent,
# which determines whether the target treats the contact as
# WELCOMED / UNWELCOMED / REFUSED / HOSTILE.
#
# Tiers:
#   10  — explicitly romantic/affectionate  (kiss, hug, …)
#   25  — gentle neutral contact            (touch, pat, tap, …)
#   70  — aggressive physical contact       (push, shove, grab, …)
#
# Verbs absent from these tables fall through to the pure freeform path.
# ---------------------------------------------------------------------------

_ROMANTIC_CONTACT_VERBS: frozenset[str] = frozenset({
    "kiss", "kiss_on_cheek", "kiss_on_forehead",
    "hug", "embrace", "cuddle", "snuggle", "nuzzle",
    "caress", "stroke", "fondle", "pet", "hold", "squeeze",
    "lean_in", "press", "nestle", "cling", "wrap",
})

_NEUTRAL_CONTACT_VERBS: frozenset[str] = frozenset({
    "touch", "tap", "pat", "poke", "prod", "nudge",
    "brush", "bump", "graze", "rub", "tickle",
    "handshake", "shake_hands", "fist_bump", "high_five",
    "bow_to", "curtsy", "salute",
})

_AGGRESSIVE_CONTACT_VERBS: frozenset[str] = frozenset({
    "push", "shove", "grab", "grab_arm", "grab_collar",
    "slap", "choke", "strangle", "tackle", "grapple",
    "restrain", "pin", "throw_down", "trip",
})


def _contact_verb_aggression(verb: str) -> Optional[int]:
    """
    Return the aggression level for `verb` if it is a physical-contact verb,
    or None if it should be handled by the pure freeform path.

    Also checks whether the verb *contains* a contact root (e.g. "gentle_kiss"
    contains "kiss") so compound verbs are caught.
    """
    v = str(verb).lower()
    if v in _ROMANTIC_CONTACT_VERBS:
        return 10
    if v in _NEUTRAL_CONTACT_VERBS:
        return 25
    if v in _AGGRESSIVE_CONTACT_VERBS:
        return 70
    # Substring matching for compound synthesised verbs like "gentle_hug",
    # "tight_squeeze", "violent_shove", "playful_pat", …
    for root in _ROMANTIC_CONTACT_VERBS:
        if root in v:
            return 10
    for root in _NEUTRAL_CONTACT_VERBS:
        if root in v:
            return 25
    for root in _AGGRESSIVE_CONTACT_VERBS:
        if root in v:
            return 70
    return None


def _emotional_for_consent(
    consent: ConsentState, current: EmotionalState
) -> EmotionalState:
    if consent == ConsentState.WELCOMED:
        return EmotionalState.FRIENDLY if current == EmotionalState.NEUTRAL else current
    if consent == ConsentState.UNWELCOMED:
        return EmotionalState.SUSPICIOUS
    if consent == ConsentState.REFUSED:
        return EmotionalState.ANGRY
    if consent == ConsentState.HOSTILE:
        return EmotionalState.HOSTILE
    return current


def _alertness_for_consent(
    consent: ConsentState, current: AlertnessLevel
) -> AlertnessLevel:
    if consent == ConsentState.WELCOMED:
        return current
    if consent == ConsentState.UNWELCOMED:
        order = [AlertnessLevel.UNAWARE, AlertnessLevel.LOW, AlertnessLevel.MEDIUM,
                 AlertnessLevel.HIGH, AlertnessLevel.COMBAT]
        idx = order.index(current) if current in order else 1
        return order[min(idx + 1, len(order) - 1)] if idx < 2 else current
    if consent == ConsentState.REFUSED:
        return (AlertnessLevel.HIGH if current != AlertnessLevel.COMBAT
                else AlertnessLevel.COMBAT)
    if consent == ConsentState.HOSTILE:
        return AlertnessLevel.COMBAT
    return current


def _compile_take(
    action: SemanticAction,
    world: WorldState,
) -> ValidationResult:
    """
    Pick up an item from an adjacent tile (or from a container in the
    actor's inventory) into the actor's loose inventory.

    Handles two cases:
    1. Item is on the ground (obj.position is not None) within distance ≤ 1.
    2. Item is inside a container the actor already holds (obj.container_id
       points to a container in actor.inventory).

    Enforces carry-capacity — the actor must have room for the item's weight.
    """
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(
            valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND
        )
    if action.target is None or isinstance(action.target, Coord):
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.MALFORMED_ACTION,
            rejection_detail="TAKE requires an object_id target.",
        )
    object_id = ObjectId(str(action.target))
    obj = grid.objects.get(object_id)
    if obj is None:
        return ValidationResult(
            valid=False, rejection_reason=RejectionReason.TARGET_NOT_FOUND
        )

    # Carry-capacity check (applies regardless of source)
    carried = current_carry_weight(actor, grid)
    if carried + obj.weight > actor.carry_capacity:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=(
                f"Cannot carry {obj.name} ({obj.weight:.1f} kg) — "
                f"already carrying {carried:.1f} kg of "
                f"{actor.carry_capacity:.1f} kg capacity."
            ),
        )

    # Case 1: item is on the ground
    if obj.position is not None:
        if actor.position.manhattan(obj.position) > 1:
            return ValidationResult(
                valid=False,
                rejection_reason=RejectionReason.TARGET_OUT_OF_RANGE,
                rejection_detail=f"{obj.name} is not adjacent.",
            )
        return ValidationResult(
            valid=True,
            concrete_transitions=[
                Transition(
                    kind=TransitionKind.ITEM_TRANSFERRED,
                    payload={
                        "object_id": object_id,
                        "from_entity": None,
                        "to_entity": action.actor,
                    },
                )
            ],
            plausibility=1.0,
        )

    # Case 2: item is inside a container the actor holds
    if obj.container_id is not None:
        container = grid.objects.get(obj.container_id)
        if container is not None and obj.container_id in actor.inventory:
            return ValidationResult(
                valid=True,
                concrete_transitions=[
                    Transition(
                        kind=TransitionKind.ITEM_RETRIEVED,
                        payload={
                            "entity_id": str(action.actor),
                            "object_id": str(object_id),
                            "container_id": str(obj.container_id),
                        },
                    )
                ],
                plausibility=1.0,
            )

    # Item is already in someone else's inventory (or orphaned)
    return ValidationResult(
        valid=False,
        rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
        rejection_detail=f"{obj.name} is not on the ground or in a container you hold.",
    )


def _compile_observe(
    action: SemanticAction,
    world: WorldState,
) -> ValidationResult:
    """
    Observe — read the world without mutating it.

    Returns a structured ``observe_result`` (tile marks, env state, durable
    facts, nearby entities) so open-ended play surfaces prior consequences.
    """
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(
            valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND
        )

    from ..fact_registry import entity_fact_lines, survey_area, survey_tile
    from ..schemas import Coord

    observe_result: dict[str, Any]

    if isinstance(action.target, Coord):
        observe_result = survey_tile(world, actor, action.target)
    elif isinstance(action.target, str):
        target = _resolve_entity_target(grid, action.target)
        if target is not None:
            observe_result = {
                "kind": "entity",
                "name": target.name,
                "distance": actor.position.manhattan(target.position),
                "emotional_state": target.emotional_state.value,
                "alertness": target.alertness.value,
                "established_facts": entity_fact_lines(world, str(target.entity_id)),
            }
        else:
            # Maybe a coord key like "3,4,0"
            parts = action.target.split(",")
            if len(parts) >= 2:
                try:
                    coord = Coord(
                        x=int(parts[0]),
                        y=int(parts[1]),
                        z=int(parts[2]) if len(parts) > 2 else 0,
                    )
                    observe_result = survey_tile(world, actor, coord)
                except ValueError:
                    observe_result = survey_area(world, actor)
            else:
                observe_result = survey_area(world, actor)
    else:
        observe_result = survey_area(world, actor)

    return ValidationResult(
        valid=True,
        concrete_transitions=[
            Transition(
                kind=TransitionKind.DIALOGUE_SPOKEN,
                payload={
                    "speaker": action.actor,
                    "listener": str(action.target) if action.target else None,
                    "content": "[observe result]",
                    "observe_result": observe_result,
                    "verb": str(action.verb),
                },
            )
        ],
        plausibility=1.0,
    )


def _compile_open_close(
    action: SemanticAction,
    world: WorldState,
) -> ValidationResult:
    grid = world.spatial
    if not isinstance(action.target, Coord):
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.MALFORMED_ACTION,
            rejection_detail="OPEN/CLOSE requires a Coord target.",
        )
    coord: Coord = action.target
    tile = grid.tile_at(coord)
    if action.verb == ActionType.OPEN:
        if tile.terrain != TerrainType.DOOR_CLOSED:
            return ValidationResult(
                valid=False,
                rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
                rejection_detail=f"No closed door at {coord}.",
            )
        return ValidationResult(
            valid=True,
            concrete_transitions=[
                Transition(
                    kind=TransitionKind.TILE_CHANGED,
                    payload={
                        "coord": {"x": coord.x, "y": coord.y},
                        "terrain": TerrainType.DOOR_OPEN.value,
                    },
                )
            ],
            plausibility=1.0,
        )
    else:
        if tile.terrain != TerrainType.DOOR_OPEN:
            return ValidationResult(
                valid=False,
                rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
                rejection_detail=f"No open door at {coord}.",
            )
        return ValidationResult(
            valid=True,
            concrete_transitions=[
                Transition(
                    kind=TransitionKind.TILE_CHANGED,
                    payload={
                        "coord": {"x": coord.x, "y": coord.y},
                        "terrain": TerrainType.DOOR_CLOSED.value,
                    },
                )
            ],
            plausibility=1.0,
        )


def _compile_symbolic(
    action: SemanticAction,
    world: WorldState,
) -> ValidationResult:
    """
    Zone 3 fallback handler for SYMBOLIC actions and orphaned WAITs.

    Produces a minimal emotional-state transition (HOPEFUL) to record that
    something happened, without asserting any physical fact.
    """
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(
            valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND
        )
    return ValidationResult(
        valid=True,
        concrete_transitions=[
            Transition(
                kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED,
                payload={
                    "entity_id": action.actor,
                    "from": actor.emotional_state.value,
                    "to": EmotionalState.NEUTRAL.value,
                    "cause": "symbolic_action",
                    "actor": action.actor,
                },
            )
        ],
        plausibility=1.0,
    )


from typing import Callable

def _compile_flee(
    action: SemanticAction,
    world: WorldState,
) -> ValidationResult:
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(
            valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND
        )

    from ..hazard_avoidance import pick_flee_coord

    best = pick_flee_coord(actor, world)
    if best is None:
        candidates = [n for n in actor.position.neighbors() if grid.is_passable(n)]
        if not candidates:
            return ValidationResult(
                valid=False,
                rejection_reason=RejectionReason.PATH_INACCESSIBLE,
                rejection_detail="No passable neighbors to flee to.",
            )
        best = candidates[0]

    payload: dict = {
        "entity_id": action.actor,
        "from": {"x": actor.position.x, "y": actor.position.y},
        "to": {"x": best.x, "y": best.y},
    }
    if actor.position.z or best.z:
        payload["from"]["z"] = actor.position.z
        payload["to"]["z"] = best.z

    from ..hazard_avoidance import movement_hazard_transitions

    hazard_extras = movement_hazard_transitions(world, action.actor, best)
    return ValidationResult(
        valid=True,
        concrete_transitions=[
            Transition(
                kind=TransitionKind.ENTITY_MOVED,
                payload=payload,
            ),
            *hazard_extras,
        ],
        plausibility=1.0,
    )


# ---------------------------------------------------------------------------
# Inventory management compilers (equip / unequip / store / retrieve)
# ---------------------------------------------------------------------------


def _compile_equip(
    action: SemanticAction,
    world: WorldState,
) -> ValidationResult:
    """
    Equip an item from the actor's inventory into a body slot.

    The item must:
    • Be in actor.inventory.
    • Declare a `slot` in its ObjectState.
    • Not have its target slot occupied by a two-handed weapon that would
      conflict (e.g. can't equip an off-hand item while wielding a
      two-handed weapon marked with tag "two_handed").

    If the slot is already occupied the existing item is automatically
    unequipped back to loose inventory (yielding a combined transition list).
    """
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)

    if action.target is None or isinstance(action.target, Coord):
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.MALFORMED_ACTION,
            rejection_detail="EQUIP requires an object_id target.",
        )
    object_id = ObjectId(str(action.target))
    obj = grid.objects.get(object_id)
    if obj is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.TARGET_NOT_FOUND)

    if object_id not in actor.inventory:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=f"{obj.name} is not in your inventory.",
        )
    if obj.slot is None:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=f"{obj.name} cannot be equipped (no equip slot defined).",
        )
    slot = obj.slot

    # Two-handed conflict: if actor wields a two-handed weapon in main_hand,
    # block equipping anything to off_hand, and vice versa.
    if slot == EquipSlot.OFF_HAND:
        mh_oid = actor.equipped_slots.get(EquipSlot.MAIN_HAND)
        if mh_oid:
            mh_obj = grid.objects.get(mh_oid)
            if mh_obj and "two_handed" in mh_obj.tags:
                return ValidationResult(
                    valid=False,
                    rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
                    rejection_detail=(
                        f"Cannot equip to off-hand while wielding "
                        f"two-handed {mh_obj.name}."
                    ),
                )
    if slot == EquipSlot.MAIN_HAND and "two_handed" in obj.tags:
        oh_oid = actor.equipped_slots.get(EquipSlot.OFF_HAND)
        if oh_oid:
            oh_obj = grid.objects.get(oh_oid)
            if oh_obj:
                return ValidationResult(
                    valid=False,
                    rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
                    rejection_detail=(
                        f"Cannot equip two-handed {obj.name} while "
                        f"{oh_obj.name} is in the off-hand. Unequip it first."
                    ),
                )

    transitions: list[Transition] = []

    # Auto-unequip whatever is in the slot already.
    replaced_oid = actor.equipped_slots.get(slot)
    if replaced_oid is not None:
        transitions.append(Transition(
            kind=TransitionKind.ITEM_UNEQUIPPED,
            payload={
                "entity_id": str(action.actor),
                "object_id": str(replaced_oid),
                "slot": slot,
            },
        ))

    transitions.append(Transition(
        kind=TransitionKind.ITEM_EQUIPPED,
        payload={
            "entity_id": str(action.actor),
            "object_id": str(object_id),
            "slot": slot,
            "replaced_object_id": str(replaced_oid) if replaced_oid else None,
        },
    ))
    return ValidationResult(valid=True, concrete_transitions=transitions, plausibility=1.0)


def _compile_unequip(
    action: SemanticAction,
    world: WorldState,
) -> ValidationResult:
    """
    Remove an item from a body slot back into loose inventory.

    The item identified by action.target must currently be in one of
    the actor's equipped_slots.
    """
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)

    if action.target is None or isinstance(action.target, Coord):
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.MALFORMED_ACTION,
            rejection_detail="UNEQUIP requires an object_id target.",
        )
    object_id = ObjectId(str(action.target))
    obj = grid.objects.get(object_id)
    if obj is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.TARGET_NOT_FOUND)

    # Find which slot it's in
    slot = next(
        (s for s, oid in actor.equipped_slots.items() if oid == object_id),
        None,
    )
    if slot is None:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=f"{obj.name} is not currently equipped.",
        )

    return ValidationResult(
        valid=True,
        concrete_transitions=[
            Transition(
                kind=TransitionKind.ITEM_UNEQUIPPED,
                payload={
                    "entity_id": str(action.actor),
                    "object_id": str(object_id),
                    "slot": slot,
                },
            )
        ],
        plausibility=1.0,
    )


def _compile_store_in(
    action: SemanticAction,
    world: WorldState,
) -> ValidationResult:
    """
    Place a loose inventory item inside a container the actor also holds.

    action.target  — the item to store (ObjectId string).
    action.intent.rationale — optionally the container name/id (e.g. "satchel").
      If omitted the engine picks the first suitable container in inventory.
    """
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)

    if action.target is None or isinstance(action.target, Coord):
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.MALFORMED_ACTION,
            rejection_detail="STORE requires an object_id target (the item to store).",
        )
    item_id = ObjectId(str(action.target))
    item = grid.objects.get(item_id)
    if item is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.TARGET_NOT_FOUND)
    if item_id not in actor.inventory:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=f"{item.name} is not in your loose inventory.",
        )
    if item.container_id is not None:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=f"{item.name} is already inside a container.",
        )

    # Resolve container: try rationale hint, then first available container.
    container: Optional[ObjectState] = None
    rationale = (action.intent.rationale or "").strip().lower() if action.intent else ""
    for c_oid in actor.inventory:
        c = grid.objects.get(c_oid)
        if c is None or not c.is_container:
            continue
        if c_oid == item_id:
            continue  # can't store a container inside itself
        if rationale and rationale not in c.name.lower():
            continue
        # Check container has enough remaining capacity
        contents_weight = sum(
            (grid.objects.get(x).weight if grid.objects.get(x) else 0.0)
            for x in c.contents
        )
        if contents_weight + item.weight > c.carry_capacity:
            continue
        contents_bulk = sum(
            (grid.objects.get(x).bulk if grid.objects.get(x) else 0.0)
            for x in c.contents
        )
        if contents_bulk + item.bulk > c.bulk_capacity:
            continue
        container = c
        break

    if container is None:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=(
                "No suitable container in your inventory "
                "(check capacity / weight / bulk limits)."
            ),
        )

    return ValidationResult(
        valid=True,
        concrete_transitions=[
            Transition(
                kind=TransitionKind.ITEM_STORED,
                payload={
                    "entity_id": str(action.actor),
                    "object_id": str(item_id),
                    "container_id": str(container.object_id),
                },
            )
        ],
        plausibility=1.0,
    )


def _compile_retrieve(
    action: SemanticAction,
    world: WorldState,
) -> ValidationResult:
    """
    Pull an item out of a container in the actor's inventory.

    action.target — the item to retrieve (ObjectId string or item name).
    Carry-capacity is checked — must have room for the item's weight.
    """
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)

    if action.target is None or isinstance(action.target, Coord):
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.MALFORMED_ACTION,
            rejection_detail="RETRIEVE requires an object_id target.",
        )
    item_id = ObjectId(str(action.target))
    item = grid.objects.get(item_id)
    if item is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.TARGET_NOT_FOUND)
    if item.container_id is None:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=f"{item.name} is not inside any container.",
        )
    container = grid.objects.get(item.container_id)
    if container is None or item.container_id not in actor.inventory:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail="Container is not in your inventory.",
        )

    # Carry-capacity check: item leaves container but stays on the entity
    # so total carry weight doesn't change — no extra check needed.
    # (The item's weight is already counted via the container path in
    # current_carry_weight.)

    return ValidationResult(
        valid=True,
        concrete_transitions=[
            Transition(
                kind=TransitionKind.ITEM_RETRIEVED,
                payload={
                    "entity_id": str(action.actor),
                    "object_id": str(item_id),
                    "container_id": str(item.container_id),
                },
            )
        ],
        plausibility=1.0,
    )



_CompilerFn = Callable[["SemanticAction", "WorldState"], "ValidationResult"]

# Populated by registry.py after all _compile_* functions are defined.
_KERNEL_COMPILERS: dict[str, _CompilerFn] = {}
_BUILTIN_DEFAULTS: dict[str, _CompilerFn] = {}

_DIR_ALIASES: dict[str, str | None] = {
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest",
    "up": "north", "down": "south",
    "right": "east", "left": "west",
    "forward": None,
    "backward": None,
    "around": None,
    "turn left": None,
    "turn right": None,
}


def _compile_turn(
    action: "SemanticAction",
    world: "WorldState",
) -> "ValidationResult":
    """
    TURN / LOOK / FACE — change the actor's facing direction.

    Target can be:
    - A cardinal string: "north", "east", "southwest", "n", "ne", …
    - A relative string: "left" (45° CCW), "right" (45° CW),
                         "around" (180°), "forward" (no-op), "backward" (180°)
    - An EntityId: turn to face that entity
    - None: keep facing (no-op, accepted silently)
    """
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(
            valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND
        )

    new_facing: Optional[FacingDirection] = None

    if action.target is None:
        # No-op: already facing wherever they face.
        return ValidationResult(valid=True, concrete_transitions=[], plausibility=1.0)

    if isinstance(action.target, Coord):
        # Face toward a coordinate.
        dx = action.target.x - actor.position.x
        dy = action.target.y - actor.position.y
        new_facing = FacingDirection.from_delta(dx, dy)
    elif isinstance(action.target, str):
        t = action.target.lower().strip()
        # Relative turns
        if t in ("left", "turn left", "ccw"):
            new_facing = actor.facing.turn(-2)   # 90° CCW (2 steps of 45°)
        elif t in ("right", "turn right", "cw"):
            new_facing = actor.facing.turn(2)    # 90° CW
        elif t in ("around", "backward", "back", "reverse"):
            new_facing = actor.facing.opposite
        elif t in ("forward", "ahead"):
            new_facing = actor.facing              # no change
        else:
            # Cardinal / alias lookup
            canonical = _DIR_ALIASES.get(t, t)
            try:
                new_facing = FacingDirection(canonical)
            except ValueError:
                # Maybe it's an entity name — face toward them
                target_ent = _resolve_entity_target(grid, action.target)
                if target_ent is not None:
                    dx = target_ent.position.x - actor.position.x
                    dy = target_ent.position.y - actor.position.y
                    new_facing = FacingDirection.from_delta(dx, dy)
                else:
                    return ValidationResult(
                        valid=False,
                        rejection_reason=RejectionReason.MALFORMED_ACTION,
                        rejection_detail=(
                            f"Cannot resolve turn target '{action.target}'. "
                            "Use a direction name (north, east, …) or an entity name."
                        ),
                    )

    if new_facing is None:
        return ValidationResult(valid=True, concrete_transitions=[], plausibility=1.0)

    if new_facing == actor.facing:
        return ValidationResult(valid=True, concrete_transitions=[], plausibility=1.0)

    return ValidationResult(
        valid=True,
        concrete_transitions=[
            Transition(
                kind=TransitionKind.ENTITY_TURNED,
                payload={
                    "entity_id": action.actor,
                    "from": actor.facing.value,
                    "to": new_facing.value,
                },
            )
        ],
        plausibility=1.0,
    )


def _compile_generic(
    action: "SemanticAction",
    world: "WorldState",
) -> "ValidationResult":
    """Delegate to the unified verb pipeline (see ``compiler/verb_pipeline.py``)."""
    from .verb_pipeline import compile_unresolved_verb

    return compile_unresolved_verb(action, world)


def _compile_freeform(
    action: "SemanticAction", world: "WorldState"
) -> "ValidationResult":
    grid = world.spatial
    target_entity: EntityState | None = None
    if isinstance(action.target, str):
        target_entity = _resolve_entity_target(grid, action.target)

    contest_outcome = _resolve_contest(action, world, target_entity)
    transitions = _validate_proposals(action.proposed_effects, world, action)

    # ── Speech loudness propagation ────────────────────────────────────────
    # When the player (or any entity) uses a loud speech verb, raise the
    # alertness of all nearby entities so the NPC policy can react this tick.
    _LOUD_VERBS: frozenset[str] = frozenset({
        "yell", "shout", "scream", "bellow", "roar", "holler", "cry",
        "announce", "declare", "exclaim", "call",
    })
    _QUIET_VERBS: frozenset[str] = frozenset({
        "whisper", "mutter", "murmur", "hiss",
    })
    verb_lower = str(action.verb).lower()
    actor_ent = grid.entities.get(action.actor)
    if actor_ent is not None:
        from ..speech_utils import SPEECH_LIKE_VERBS

        if verb_lower in _LOUD_VERBS:
            # Loud: radius 8, always raises to at least MEDIUM
            _noise_radius = 8
            _min_alertness = AlertnessLevel.MEDIUM
        elif verb_lower in SPEECH_LIKE_VERBS:
            # Normal speech: radius 4, raises UNAWARE→LOW
            _noise_radius = 4
            _min_alertness = AlertnessLevel.LOW
        elif verb_lower in _QUIET_VERBS:
            _noise_radius = 0
            _min_alertness = None
        else:
            _noise_radius = 0
            _min_alertness = None

        if _min_alertness is not None:
            from ..speech_utils import best_speech_line, sanitize_dialogue_text

            spoken_text = sanitize_dialogue_text(best_speech_line(action))
            # Emit a single DIALOGUE_SPOKEN with loudness metadata first
            transitions.insert(0, Transition(
                kind=TransitionKind.DIALOGUE_SPOKEN,
                payload={
                    "actor": action.actor,
                    "target": str(action.target) if isinstance(action.target, str) else None,
                    "verb": verb_lower,
                    "text": spoken_text,
                    "loud": verb_lower in _LOUD_VERBS,
                    "noise_level": 90 if verb_lower in _LOUD_VERBS else 40,
                    "source_pos": {"x": actor_ent.position.x, "y": actor_ent.position.y},
                },
            ))
            # ── Claim / threat detection ─────────────────────────────────
            # Analyse the spoken text for declarative claims that change world
            # state (threats, boasts, confessions, hostile statements).  This
            # is intentionally generic — we detect *any* bold statement, not
            # just combat threats, so the system can handle rumour-spreading,
            # social pressure, diplomatic claims, etc.
            _claim_type, _severity = _classify_claim(spoken_text)
            if _claim_type and spoken_text:
                # Collect entities named or implied in the claim
                _claim_targets: list[str] = []
                if action.target and isinstance(action.target, str):
                    _claim_targets.append(str(action.target))
                # Also record actor's current targets visible on map
                for _eid2, _ent2 in grid.entities.items():
                    if _eid2 == action.actor:
                        continue
                    _dist2 = actor_ent.position.manhattan(_ent2.position)
                    if _dist2 <= _noise_radius:
                        if _eid2 not in _claim_targets:
                            _claim_targets.append(_eid2)
                transitions.append(Transition(
                    kind=TransitionKind.CLAIM_MADE,
                    payload={
                        "actor": action.actor,
                        "text": spoken_text,
                        "claim_type": _claim_type,
                        "severity": _severity,
                        "targets": _claim_targets,
                    },
                ))
                # Threatening claims spike alertness of ALL witnesses to HIGH
                if _severity >= 0.6:
                    _threat_level = AlertnessLevel.HIGH
                elif _severity >= 0.3:
                    _threat_level = AlertnessLevel.MEDIUM
                else:
                    _threat_level = None
                if _threat_level is not None:
                    _min_alertness = _threat_level  # override for loop below
            # ── End claim detection ──────────────────────────────────────

            # Raise alertness of all entities in radius
            for eid, ent in grid.entities.items():
                if eid == action.actor:
                    continue
                dist = actor_ent.position.manhattan(ent.position)
                if dist > _noise_radius:
                    continue
                # Don't lower alertness — only raise it
                current_level = list(AlertnessLevel)
                try:
                    cur_idx = current_level.index(ent.alertness)
                    min_idx = current_level.index(_min_alertness)
                except ValueError:
                    continue
                if cur_idx < min_idx:
                    transitions.append(Transition(
                        kind=TransitionKind.ENTITY_ALERTNESS_CHANGED,
                        payload={
                            "entity_id": eid,
                            "from": ent.alertness.value,
                            "to": _min_alertness.value,
                            "cause": f"heard_{verb_lower}",
                            "source": str(action.actor),
                        },
                    ))
            # Skip the duplicate _interaction_record DIALOGUE_SPOKEN below
            return ValidationResult(
                valid=True,
                concrete_transitions=transitions,
                plausibility=0.9 if contest_outcome == "success" else 0.5,
            )
    # ── End speech propagation ─────────────────────────────────────────────

    transitions.append(_interaction_record(action, contest_outcome))

    return ValidationResult(
        valid=True,
        concrete_transitions=transitions,
        plausibility=0.9 if contest_outcome == "success" else 0.5,
    )


def _compile_from_template(
    action: "SemanticAction",
    world: "WorldState",
    template: "VerbTemplate",
) -> "ValidationResult":
    from .effects import compile_effects

    return compile_effects(action, world, template=template)


def _resolve_contest(
    action: "SemanticAction",
    world: "WorldState",
    target_entity: EntityState | None,
) -> str:
    """Return 'success' or 'failure' for action.contest, or 'success' if
    no contest is specified or no target is available."""
    grid = _grid_for_action(world, action)
    actor = grid.entities.get(action.actor)
    contest = action.contest
    if contest is None or actor is None:
        return "success"
    actor_score = actor.attributes.get(contest.actor_attribute, 50)
    if target_entity is not None and contest.target_attribute:
        target_score = target_entity.attributes.get(
            contest.target_attribute, 50
        )
        if target_entity.alertness == AlertnessLevel.HIGH:
            target_score += 15
        elif target_entity.alertness == AlertnessLevel.UNAWARE:
            target_score -= 10
    else:
        target_score = contest.difficulty
    if action.style.aggression > 70:
        actor_score += 10
    seed = f"{action.action_id}_{world.tick}_contest"
    return "success" if _contest_roll(actor_score, target_score, seed) else "failure"


def _substitution_table(
    action: "SemanticAction", target_entity: EntityState | None
) -> dict[str, Any]:
    return {
        "$actor": action.actor,
        "$target": (
            target_entity.entity_id if target_entity
            else (str(action.target) if isinstance(action.target, str) else None)
        ),
    }


def _classify_claim(text: str) -> tuple[str, float]:
    """
    Analyse a spoken sentence and return (claim_type, severity 0-1).

    This is intentionally broad — we don't just look for combat threats.
    Any bold declarative statement changes social state.

    Returns ("", 0.0) if the text is mundane.

    claim_type values (open-ended, not exhaustive):
      "threat"      — declares intent to harm someone/something
      "boast"       — claim of power/ability/status
      "insult"      — direct social attack
      "confession"  — admission of wrongdoing
      "vow"         — solemn promise
      "accusation"  — blames another party
      "rumour"      — claims knowledge of hidden information
      "declaration" — general provocative assertion
    """
    if not text:
        return "", 0.0
    lower = text.lower()

    # Threat patterns (high severity)
    _THREAT_WORDS = {
        "kill", "murder", "assassinate", "slay", "execute", "destroy",
        "burn", "annihilate", "obliterate", "end you", "end him", "end her",
        "cut down", "gut", "hang", "behead", "poison", "crush",
        "will die", "shall die", "you are dead", "you're dead",
    }
    if any(w in lower for w in _THREAT_WORDS):
        return "threat", 0.8

    # Insults (medium-high severity)
    _INSULT_WORDS = {
        "bastard", "coward", "traitor", "fool", "idiot", "worthless",
        "pig", "dog", "scum", "filth", "vermin", "worm", "rat",
        "liar", "cheat", "thief", "deceiver", "corrupt",
    }
    if any(w in lower for w in _INSULT_WORDS):
        return "insult", 0.55

    # Boasts / declarations of power
    _BOAST_WORDS = {
        "i am the", "i am a", "bow before", "kneel", "fear me",
        "you cannot stop", "none can stop", "i will rule",
        "power beyond", "unstoppable",
    }
    if any(w in lower for w in _BOAST_WORDS):
        return "boast", 0.4

    # Confessions
    _CONFESS_WORDS = {
        "i killed", "i stole", "i burned", "i poisoned", "i betrayed",
        "i murdered", "i am guilty", "i did it", "it was me",
        "i lied", "i cheated",
    }
    if any(w in lower for w in _CONFESS_WORDS):
        return "confession", 0.65

    # Vows
    _VOW_WORDS = {
        "i swear", "i vow", "i promise", "upon my honour", "on my life",
        "i pledge", "by the gods", "i give my word",
    }
    if any(w in lower for w in _VOW_WORDS):
        return "vow", 0.25

    # Accusations
    _ACCUSE_WORDS = {
        "is a traitor", "is guilty", "is corrupt", "is lying",
        "he killed", "she killed", "they killed", "conspired",
        "is behind", "is responsible", "sold us out",
    }
    if any(w in lower for w in _ACCUSE_WORDS):
        return "accusation", 0.5

    return "", 0.0


def _interaction_record(action: "SemanticAction", outcome: str) -> "Transition":
    """The catch-all transition that records that the verb happened.

    Stored as DIALOGUE_SPOKEN so the existing narrator path can pick it
    up; the payload carries the verb, manner, target, and contest
    outcome so downstream consumers can render or analyze faithfully.
    """
    target_str: Any = None
    if isinstance(action.target, str):
        target_str = action.target
    return Transition(
        kind=TransitionKind.DIALOGUE_SPOKEN,
        payload={
            "actor": action.actor,
            "target": target_str,
            "verb": action.verb,
            "manner": action.intent.manner if action.intent else None,
            "contest_outcome": outcome,
            "text": (
                action.intent.manner
                if action.intent and action.intent.manner else None
            ),
        },
    )


# LM consequence proposer often says "kind" instead of "edge_kind", or uses
# colloquial labels — map to canonical EdgeKind values before graph apply.

def _parse_relationship_spec(
    action: SemanticAction,
) -> tuple[str, Optional[float], Optional[float]]:
    """
    Parse edge_kind, optional absolute weight, optional delta from intent fields.

    Accepts manner/rationale like ``distrusts``, ``ally_of:0.6``, or
    ``delta:0.15`` in desired_outcome.
    """
    import re as _re_rel

    parts: list[str] = []
    if action.intent:
        if action.intent.manner:
            parts.append(action.intent.manner)
        if action.intent.rationale:
            parts.append(action.intent.rationale)
        for item in action.intent.desired_outcome or []:
            parts.append(str(item))
    text = " ".join(parts).lower().strip()

    weight: Optional[float] = None
    delta: Optional[float] = None
    wm = _re_rel.search(r"(?:weight|w)\s*[:=]\s*(-?\d+(?:\.\d+)?)", text)
    if wm:
        weight = float(wm.group(1))
    dm = _re_rel.search(r"delta\s*[:=]\s*(-?\d+(?:\.\d+)?)", text)
    if dm:
        delta = float(dm.group(1))

    edge_kind = EdgeKind.INTERACTED.value
    for token in _re_rel.split(r"[\s,;|]+", text):
        if not token or token in ("delta", "weight", "edge", "kind", "add", "update"):
            continue
        if _re_rel.match(r"^-?\d", token):
            continue
        if ":" in token:
            token = token.split(":", 1)[0]
        key = token.strip()
        if key in _EDGE_KIND_ALIASES:
            edge_kind = _EDGE_KIND_ALIASES[key]
            break
        try:
            edge_kind = EdgeKind(key).value
            break
        except ValueError:
            continue
    return edge_kind, weight, delta


def _compile_edge_add(
    action: SemanticAction,
    world: WorldState,
) -> ValidationResult:
    """Create or replace a directed relational edge actor → target."""
    grid = world.spatial
    if action.target is None or isinstance(action.target, Coord):
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.MALFORMED_ACTION,
            rejection_detail="edge_add requires an entity target.",
        )
    target_id = str(action.target)
    if EntityId(target_id) not in grid.entities:
        return ValidationResult(
            valid=False, rejection_reason=RejectionReason.TARGET_NOT_FOUND
        )
    edge_kind, weight, _delta = _parse_relationship_spec(action)
    w = 0.5 if weight is None else max(-1.0, min(1.0, float(weight)))
    actor_id = str(action.actor)
    transitions = [
        Transition(
            kind=TransitionKind.EDGE_CREATED,
            payload={
                "source": actor_id,
                "target": target_id,
                "edge_kind": edge_kind,
                "weight": w,
                "meta": {"verb": action.verb, "cause": "edge_add"},
            },
        ),
        _interaction_record(action, "success"),
    ]
    return ValidationResult(valid=True, concrete_transitions=transitions, plausibility=0.9)


def _compile_edge_update(
    action: SemanticAction,
    world: WorldState,
) -> ValidationResult:
    """Adjust weight on an existing edge (or no-op if missing)."""
    grid = world.spatial
    if action.target is None or isinstance(action.target, Coord):
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.MALFORMED_ACTION,
            rejection_detail="edge_update requires an entity target.",
        )
    target_id = str(action.target)
    if EntityId(target_id) not in grid.entities:
        return ValidationResult(
            valid=False, rejection_reason=RejectionReason.TARGET_NOT_FOUND
        )
    edge_kind, _weight, delta = _parse_relationship_spec(action)
    d = 0.12 if delta is None else max(-0.5, min(0.5, float(delta)))
    actor_id = str(action.actor)
    transitions = [
        Transition(
            kind=TransitionKind.EDGE_UPDATED,
            payload={
                "source": actor_id,
                "target": target_id,
                "edge_kind": edge_kind,
                "delta": d,
                "meta": {"verb": action.verb, "cause": "edge_update"},
            },
        ),
        _interaction_record(action, "success"),
    ]
    return ValidationResult(valid=True, concrete_transitions=transitions, plausibility=0.9)


def _compile_wait(action: SemanticAction, world: WorldState) -> ValidationResult:
    """
    WAIT — pass time in place: rest fatigue, calm alertness, extend stealth.
    """
    from ..needs import satisfy_need

    actor = world.spatial.entities.get(action.actor)
    if actor is None:
        return ValidationResult(
            valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND
        )

    actor_id = str(action.actor)
    transitions: list[Transition] = [
        satisfy_need(actor_id, "fatigue", 12.0, cause="wait"),
        satisfy_need(actor_id, "purpose", 3.0, cause="wait"),
    ]

    if actor.meta.get("hidden"):
        transitions.append(
            Transition(
                kind=TransitionKind.ENTITY_PROPERTY_CHANGED,
                payload={
                    "entity_id": actor_id,
                    "prop": "hidden_ticks",
                    "new_value": 0,
                },
            )
        )

    if actor.alertness in (AlertnessLevel.HIGH, AlertnessLevel.COMBAT):
        transitions.append(
            Transition(
                kind=TransitionKind.ENTITY_ALERTNESS_CHANGED,
                payload={
                    "entity_id": actor_id,
                    "from": actor.alertness.value,
                    "to": AlertnessLevel.MEDIUM.value,
                    "cause": "wait",
                },
            )
        )

    return ValidationResult(valid=True, concrete_transitions=transitions, plausibility=1.0)


def _nearest_cover_coord(
    grid: SpatialGrid,
    origin: Coord,
    *,
    max_radius: int = 4,
) -> Optional[Coord]:
    """Return a passable tile near cover (objects, cover tags, or corners)."""
    best: Optional[tuple[tuple[int, int], Coord]] = None
    for radius in range(1, max_radius + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) + abs(dy) != radius:
                    continue
                c = Coord(x=origin.x + dx, y=origin.y + dy, z=origin.z)
                if not grid.is_in_bounds(c) or not grid.is_passable(c):
                    continue
                tile = grid.tile_at(c)
                score = 0
                if "cover" in tile.tags:
                    score += 4
                for obj in grid.objects.values():
                    if obj.position == c and not obj.passable:
                        score += 3
                    elif obj.position.manhattan(c) == 1 and not obj.passable:
                        score += 2
                open_n = sum(
                    1 for n in c.neighbors()
                    if grid.is_in_bounds(n) and grid.is_passable(n)
                )
                if open_n <= 2:
                    score += 1
                if score <= 0:
                    continue
                dist = origin.manhattan(c)
                rank = (score, -dist)
                if best is None or rank > best[0]:
                    best = (rank, c)
        if best is not None:
            return best[1]
    return None


def _compile_hide(action: SemanticAction, world: WorldState) -> ValidationResult:
    """
    HIDE — seek cover, contest stealth vs nearby observers, set hidden meta.
    """
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(
            valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND
        )

    actor_id = str(action.actor)
    transitions: list[Transition] = []
    hide_pos = actor.position

    cover = _nearest_cover_coord(grid, actor.position)
    if cover is not None and cover != actor.position:
        hide_pos = cover
        transitions.append(
            Transition(
                kind=TransitionKind.ENTITY_MOVED,
                payload={
                    "entity_id": actor_id,
                    "from": {"x": actor.position.x, "y": actor.position.y},
                    "to": {"x": cover.x, "y": cover.y},
                },
            )
        )

    actor_stealth = int(actor.attributes.get("stealth", 50))
    detected_by: list[str] = []

    for observer in grid.entities.values():
        if observer.entity_id == action.actor or not observer.alive:
            continue
        if observer.position.manhattan(hide_pos) > observer.sight_range:
            continue
        visible = visible_from(
            grid,
            observer.position,
            observer.sight_range,
            facing=observer.facing,
            fov_degrees=observer.fov_degrees,
        )
        if hide_pos not in visible:
            continue

        alertness_bonus = {
            "unaware": -15, "low": -5, "medium": 0, "high": 15, "combat": 25,
        }.get(observer.alertness.value, 0)
        dist = observer.position.manhattan(hide_pos)
        target_score = (
            int(observer.attributes.get("perception", 50))
            + alertness_bonus
            + dist * 4
        )
        seed = f"{action.action_id}_{world.tick}_hide_{observer.entity_id}"
        if not _contest_roll(actor_stealth + 8, target_score, seed):
            detected_by.append(str(observer.entity_id))
            transitions.append(
                Transition(
                    kind=TransitionKind.ENTITY_ALERTNESS_CHANGED,
                    payload={
                        "entity_id": str(observer.entity_id),
                        "from": observer.alertness.value,
                        "to": AlertnessLevel.HIGH.value,
                        "cause": "spotted_hide_attempt",
                    },
                )
            )

    hidden = len(detected_by) == 0
    transitions.append(
        Transition(
            kind=TransitionKind.ENTITY_PROPERTY_CHANGED,
            payload={
                "entity_id": actor_id,
                "prop": "hidden",
                "new_value": hidden,
            },
        )
    )
    transitions.append(
        Transition(
            kind=TransitionKind.ENTITY_PROPERTY_CHANGED,
            payload={
                "entity_id": actor_id,
                "prop": "hidden_ticks",
                "new_value": 0,
            },
        )
    )
    transitions.append(
        Transition(
            kind=TransitionKind.ENTITY_ALERTNESS_CHANGED,
            payload={
                "entity_id": actor_id,
                "from": actor.alertness.value,
                "to": AlertnessLevel.LOW.value,
                "cause": "hide",
            },
        )
    )

    plausibility = 1.0 if hidden else 0.55
    return ValidationResult(
        valid=True, concrete_transitions=transitions, plausibility=plausibility
    )


def _compile_cast(
    action: "SemanticAction",
    world: "WorldState",
) -> "ValidationResult":
    """CAST kernel: routes [cast] programs to the spell runtime."""
    from ..spell_runtime import SpellError, compile_spell_text
    from ..spell_types import SpellVocabulary

    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(
            valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND
        )

    vocab: SpellVocabulary | None = world.config.spell_vocab
    if vocab is None:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.ACTION_TYPE_UNSUPPORTED,
            rejection_detail="Magic is not available in this world.",
        )

    source = (action.intent.rationale or "").strip()
    if not source:
        raw_target = str(action.target) if action.target else ""
        spell_name = (action.intent.manner or raw_target or "").strip()
        source = actor.inscribed_spells.get(spell_name, "")
        if not source:
            return ValidationResult(
                valid=False,
                rejection_reason=RejectionReason.MALFORMED_ACTION,
                rejection_detail=(
                    f"No spell program provided and no inscribed spell "
                    f"named '{spell_name}'."
                ),
            )

    unknown = _unknown_ops_in_source(source, actor.known_spell_ops, vocab)
    if unknown:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=(
                f"Unknown spell operation(s): {', '.join(sorted(unknown))}. "
                "Read a grimoire to learn these words before casting."
            ),
        )

    target_id = (
        EntityId(str(action.target))
        if isinstance(action.target, str) and action.target in grid.entities
        else None
    )

    try:
        transitions = compile_spell_text(
            source=source,
            actor_id=action.actor,
            target_id=target_id,
            world=world,
            vocab=vocab,
        )
    except SpellError as exc:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=f"Spell error: {exc}",
        )

    return ValidationResult(
        valid=True,
        concrete_transitions=transitions,
        plausibility=1.0,
    )


def _unknown_ops_in_source(
    source: str,
    known_ops: list[str],
    vocab: "Any",
) -> set[str]:
    """Lexical scan for spell ops the actor does not know."""
    import re

    known_upper = {op.upper() for op in known_ops}
    all_vocab_ops = set(vocab.operations.keys())
    _LANG_KEYWORDS = frozenset({
        "COST", "FOR", "EACH", "IF", "ELSE", "REPEAT", "PERSIST",
        "SELF", "TARGET", "NEAR", "LINE", "ALL", "TAGGED",
        "HAS_TAG", "PROP", "NOT", "ON",
        "MANA",
    })

    unknown: set[str] = set()
    for tok in re.findall(r"\b[A-Z][A-Z_0-9]*\b", source):
        if tok in _LANG_KEYWORDS:
            continue
        if tok in all_vocab_ops:
            if tok not in known_upper:
                unknown.add(tok)
        elif tok not in known_upper:
            unknown.add(tok)
    return unknown

