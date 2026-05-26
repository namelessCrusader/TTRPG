"""
Kernel compilers for embodied interactions that need real physics,
not just verb-template effects.

Each verb here:
  • Validates presence (target exists, in reach)
  • Emits typed Transitions that mutate canonical state
  • Composes with consequences.finalize_compiled_result (sticky + trace)

Verbs covered:
  posture     — sit, stand, lean, slouch, kneel
  hygiene     — wipe, clean, relieve, wash
  sustenance  — eat, sip (lighter than drink)
  perception  — sniff, listen
  expressive  — laugh, sing, hum (mood ripple to nearby)
  contact     — pat, hug, kiss, slap, spit, shove, nudge (CONTACT_INITIATED)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .schemas import (
    ActionType,
    AlertnessLevel,
    Coord,
    EdgeKind,
    EmotionalState,
    EntityId,
    EntityState,
    IntentBlock,
    ObjectId,
    ObjectState,
    RejectionReason,
    SemanticAction,
    StyleBlock,
    Transition,
    TransitionKind,
    ValidationResult,
)

if TYPE_CHECKING:
    from .schemas import WorldState


# ── Helpers ──────────────────────────────────────────────────────────────


def _actor(action: SemanticAction, world: "WorldState") -> Optional[EntityState]:
    return world.spatial.entities.get(action.actor)


def _resolve_object(world: "WorldState", raw: str) -> Optional[ObjectState]:
    oid = ObjectId(str(raw))
    return world.spatial.objects.get(oid)


def _resolve_entity(world: "WorldState", raw: str) -> Optional[EntityState]:
    eid = EntityId(str(raw))
    ent = world.spatial.entities.get(eid)
    if ent is not None:
        return ent
    lower = str(raw).lower()
    for e in world.spatial.entities.values():
        if e.name.lower() == lower:
            return e
    return None


def _nearby_entities(
    world: "WorldState",
    center: Coord,
    *,
    radius: int = 3,
    exclude: Optional[EntityId] = None,
) -> list[EntityState]:
    out: list[EntityState] = []
    for ent in world.spatial.entities.values():
        if not ent.alive:
            continue
        if exclude and ent.entity_id == exclude:
            continue
        if ent.position.manhattan(center) <= radius:
            out.append(ent)
    return out


def _set_posture(actor: EntityState, posture: str) -> Transition:
    return Transition(
        kind=TransitionKind.ENTITY_PROPERTY_CHANGED,
        payload={
            "entity_id": str(actor.entity_id),
            "prop": "posture",
            "new_value": posture,
            "cause": "posture_change",
        },
    )


# ── Posture verbs ────────────────────────────────────────────────────────


def _compile_sit(action: SemanticAction, world: "WorldState") -> ValidationResult:
    actor = _actor(action, world)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)
    if actor.meta.get("posture") == "sitting":
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail="Already seated.",
        )
    transitions: list[Transition] = [_set_posture(actor, "sitting")]
    # Sitting is restful — small stamina + fatigue need recovery.
    transitions.append(Transition(
        kind=TransitionKind.NEED_CHANGED,
        payload={
            "entity_id": str(actor.entity_id),
            "need": "fatigue",
            "delta": 3.0,
            "cause": "sit",
        },
    ))
    return ValidationResult(valid=True, concrete_transitions=transitions, plausibility=1.0)


def _compile_stand(action: SemanticAction, world: "WorldState") -> ValidationResult:
    actor = _actor(action, world)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)
    current = actor.meta.get("posture", "standing")
    if current == "standing":
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail="Already standing.",
        )
    return ValidationResult(
        valid=True,
        concrete_transitions=[_set_posture(actor, "standing")],
        plausibility=1.0,
    )


def _compile_lean(action: SemanticAction, world: "WorldState") -> ValidationResult:
    actor = _actor(action, world)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)
    return ValidationResult(
        valid=True,
        concrete_transitions=[_set_posture(actor, "leaning")],
        plausibility=1.0,
    )


def _compile_kneel(action: SemanticAction, world: "WorldState") -> ValidationResult:
    actor = _actor(action, world)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)
    return ValidationResult(
        valid=True,
        concrete_transitions=[_set_posture(actor, "kneeling")],
        plausibility=1.0,
    )


# ── Hygiene ──────────────────────────────────────────────────────────────


def _compile_wipe(action: SemanticAction, world: "WorldState") -> ValidationResult:
    """Wipe an adjacent wet tile or object, removing the wet state."""
    actor = _actor(action, world)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)
    grid = world.spatial

    target_obj: Optional[ObjectState] = None
    target_coord: Optional[Coord] = None

    if action.target:
        obj = _resolve_object(world, str(action.target))
        if obj is not None:
            target_obj = obj
            target_coord = obj.position

    if target_obj is None and target_coord is None:
        from .fluids import tile_is_wet as _tile_is_wet
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                c = Coord(x=actor.position.x + dx, y=actor.position.y + dy)
                if not grid.is_in_bounds(c):
                    continue
                if _tile_is_wet(grid, c):
                    target_coord = c
                    break
            if target_coord:
                break

    if target_obj is None and target_coord is None:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.TARGET_NOT_FOUND,
            rejection_detail="Nothing nearby to wipe.",
        )

    transitions: list[Transition] = []
    if target_coord is not None:
        transitions.append(Transition(
            kind=TransitionKind.FLUID_CHANGED,
            payload={
                "target_kind": "tile_dry",
                "x": target_coord.x,
                "y": target_coord.y,
                "cause": "wipe",
            },
        ))
        # Clear every ground-layer substance from the wiped tile.  We
        # union the canonical field layer with the legacy ``contamination``
        # mirror (still emitted by some legacy code paths) so the wipe
        # action remains complete during the fields migration.
        tile = grid.tile_at(target_coord)
        ground_layer = (tile.env.get("fields") or {}).get("ground") or {}
        legacy_cont = tile.env.get("contamination") or {}
        substance_ids = {
            str(k).lower() for k in list(ground_layer.keys()) + list(legacy_cont.keys())
        }
        for substance_id in substance_ids:
            transitions.append(Transition(
                kind=TransitionKind.FIELD_CHANGED,
                payload={
                    "medium": "tile.ground",
                    "substance": substance_id,
                    "operation": "clear",
                    "amount": 0.0,
                    "x": target_coord.x,
                    "y": target_coord.y,
                    "cause": "wipe",
                    "tick": world.tick,
                },
            ))
    if target_obj is not None and "wet" in target_obj.tags:
        transitions.append(Transition(
            kind=TransitionKind.STRUCTURE_MODIFIED,
            payload={
                "object_id": str(target_obj.object_id),
                "x": target_obj.position.x if target_obj.position else 0,
                "y": target_obj.position.y if target_obj.position else 0,
                "remove_tags": ["wet", "sticky"],
                "add_tags": [],
                "cause": "wiped",
            },
        ))
    return ValidationResult(valid=True, concrete_transitions=transitions, plausibility=1.0)


def _compile_clean(action: SemanticAction, world: "WorldState") -> ValidationResult:
    """Clean an object — removes 'dirty' tag, raises durability slightly."""
    actor = _actor(action, world)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)
    if not action.target:
        return _compile_wipe(action, world)
    obj = _resolve_object(world, str(action.target))
    if obj is None or obj.position is None:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.TARGET_NOT_FOUND,
            rejection_detail="Nothing to clean.",
        )
    if actor.position.manhattan(obj.position) > 2:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail="Too far to clean that.",
        )
    transitions = [Transition(
        kind=TransitionKind.STRUCTURE_MODIFIED,
        payload={
            "object_id": str(obj.object_id),
            "x": obj.position.x,
            "y": obj.position.y,
            "remove_tags": ["dirty", "stained", "wet"],
            "add_tags": ["clean"],
            "cause": "cleaned",
        },
    )]
    # Also clear field substances on the tile the object sits on.
    tile = grid.tile_at(obj.position)
    for layer_key in ("ground",):
        for substance_id in list(tile.env.get("fields", {}).get(layer_key, {}).keys()):
            transitions.append(Transition(
                kind=TransitionKind.FIELD_CHANGED,
                payload={
                    "medium": f"tile.{layer_key}",
                    "substance": substance_id,
                    "operation": "clear",
                    "amount": 0.0,
                    "x": obj.position.x,
                    "y": obj.position.y,
                    "cause": "clean",
                    "tick": world.tick,
                },
            ))
    return ValidationResult(valid=True, concrete_transitions=transitions, plausibility=1.0)


def _compile_relieve(action: SemanticAction, world: "WorldState") -> ValidationResult:
    """Relieve oneself — clears bladder, requires privy / outdoors tile / privacy."""
    actor = _actor(action, world)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)

    grid = world.spatial
    tile = grid.tile_at(actor.position)
    privy_tags = {"privy", "outdoors", "alley", "private"}
    has_privy = bool(privy_tags.intersection({t.lower() for t in tile.tags}))
    if not has_privy:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                t = grid.tile_at(Coord(x=actor.position.x + dx, y=actor.position.y + dy))
                if privy_tags.intersection({s.lower() for s in t.tags}):
                    has_privy = True
                    break
            if has_privy:
                break
    # An adjacent privy object also satisfies privacy.
    if not has_privy:
        for obj in grid.objects.values():
            if obj.position is None:
                continue
            otags = {t.lower() for t in obj.tags}
            if privy_tags.intersection(otags) and actor.position.manhattan(obj.position) <= 2:
                has_privy = True
                break

    bladder = float(actor.stats.get("bladder", 0))
    transitions: list[Transition] = [
        Transition(
            kind=TransitionKind.NEED_CHANGED,
            payload={
                "entity_id": str(actor.entity_id),
                "need": "bladder",
                "delta": -(bladder - 5.0),
                "cause": "relieve",
            },
        ),
    ]
    from .contamination import biofluid_deposit_from_relief

    transitions.extend(
        biofluid_deposit_from_relief(world, actor, privy=has_privy)
    )
    if not has_privy:
        transitions.append(Transition(
            kind=TransitionKind.FLUID_CHANGED,
            payload={
                "target_kind": "entity",
                "entity_id": str(actor.entity_id),
                "material": "urine",
                "volume_ml": 80.0,
                "cause": "public_relief",
                "wet": True,
            },
        ))
        transitions.append(Transition(
            kind=TransitionKind.ENTITY_STAT_CHANGED,
            payload={
                "entity_id": str(actor.entity_id),
                "stat": "reputation",
                "delta": -8.0,
                "cause": "public_relief",
            },
        ))
        transitions.append(Transition(
            kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED,
            payload={
                "entity_id": str(actor.entity_id),
                "to": EmotionalState.HUMILIATED.value,
            },
        ))
    return ValidationResult(valid=True, concrete_transitions=transitions, plausibility=1.0)


# ── Sustenance ───────────────────────────────────────────────────────────


def _find_food_for(actor: EntityState, world: "WorldState") -> Optional[ObjectState]:
    grid = world.spatial
    # Inventory first.
    for oid in actor.inventory:
        obj = grid.objects.get(oid)
        if obj is None:
            continue
        if "food" in obj.tags or "edible" in obj.tags:
            return obj
    # Adjacent tile / plate.
    for obj in grid.objects.values():
        if obj.position is None:
            continue
        if actor.position.manhattan(obj.position) <= 1 and (
            "food" in obj.tags or "edible" in obj.tags
        ):
            return obj
    return None


def _compile_eat(action: SemanticAction, world: "WorldState") -> ValidationResult:
    actor = _actor(action, world)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)
    food: Optional[ObjectState] = None
    if action.target:
        obj = _resolve_object(world, str(action.target))
        if obj is not None and ("food" in obj.tags or "edible" in obj.tags):
            food = obj
    if food is None:
        food = _find_food_for(actor, world)
    if food is None:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.TARGET_NOT_FOUND,
            rejection_detail="No food in reach.",
        )

    bites = max(1, int(food.attributes.get("bites", 3)) - 1)
    nutrition = float(food.attributes.get("nutrition", 12))
    transitions: list[Transition] = [
        Transition(
            kind=TransitionKind.NEED_CHANGED,
            payload={
                "entity_id": str(actor.entity_id),
                "need": "hunger",
                "delta": nutrition,
                "cause": "eat",
            },
        ),
    ]
    if bites <= 0:
        transitions.append(Transition(
            kind=TransitionKind.ITEM_DESTROYED,
            payload={"object_id": str(food.object_id), "cause": "consumed"},
        ))
    else:
        # Track remaining bites so subsequent eats deplete it.
        food.attributes["bites"] = bites
    return ValidationResult(valid=True, concrete_transitions=transitions, plausibility=1.0)


# ── Perception ───────────────────────────────────────────────────────────


def _compile_sniff(action: SemanticAction, world: "WorldState") -> ValidationResult:
    """Sniff — adds an observation memory tag, raises perception sharpness."""
    actor = _actor(action, world)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)
    grid = world.spatial
    smells: list[str] = []
    for ent in _nearby_entities(world, actor.position, radius=3, exclude=actor.entity_id):
        if "wet" in ent.tags:
            smells.append(f"{ent.name} smells damp")
        if "on_fire" in ent.tags:
            smells.append(f"{ent.name} smells of smoke")
    for obj in grid.objects.values():
        if obj.position is None or actor.position.manhattan(obj.position) > 3:
            continue
        if "alcohol" in obj.tags or "drink" in obj.tags:
            smells.append(f"{obj.name} — sharp scent of ale")
        if "on_fire" in obj.tags:
            smells.append(f"{obj.name} — burning")
        if "food" in obj.tags:
            smells.append(f"{obj.name} — savoury")
    summary = "; ".join(smells[:3]) or "nothing notable"
    transitions = [Transition(
        kind=TransitionKind.ENTITY_PROPERTY_CHANGED,
        payload={
            "entity_id": str(actor.entity_id),
            "prop": "last_smell",
            "new_value": summary,
            "cause": "sniff",
        },
    )]
    return ValidationResult(valid=True, concrete_transitions=transitions, plausibility=1.0)


def _compile_listen(action: SemanticAction, world: "WorldState") -> ValidationResult:
    actor = _actor(action, world)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)
    transitions = [
        Transition(
            kind=TransitionKind.ENTITY_ALERTNESS_CHANGED,
            payload={
                "entity_id": str(actor.entity_id),
                "to": AlertnessLevel.MEDIUM.value,
                "from": actor.alertness.value,
            },
        ),
        Transition(
            kind=TransitionKind.ENTITY_PROPERTY_CHANGED,
            payload={
                "entity_id": str(actor.entity_id),
                "prop": "listening",
                "new_value": world.tick,
                "cause": "listen",
            },
        ),
    ]
    return ValidationResult(valid=True, concrete_transitions=transitions, plausibility=1.0)


# ── Expressive (mood ripple) ─────────────────────────────────────────────


def _compile_laugh(action: SemanticAction, world: "WorldState") -> ValidationResult:
    return _ripple_mood(action, world, EmotionalState.HAPPY, radius=4)


def _compile_sing(action: SemanticAction, world: "WorldState") -> ValidationResult:
    actor = _actor(action, world)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)
    # Singing makes a noise event and lifts mood in larger radius.
    transitions: list[Transition] = [
        Transition(
            kind=TransitionKind.NOISE_EVENT,
            payload={
                "source_id": str(actor.entity_id),
                "source_pos": {"x": actor.position.x, "y": actor.position.y},
                "noise_level": 35,
                "description": f"{actor.name} sings",
            },
        ),
    ]
    transitions.extend(_mood_ripple_transitions(actor, world, EmotionalState.HAPPY, radius=5))
    return ValidationResult(valid=True, concrete_transitions=transitions, plausibility=1.0)


def _compile_hum(action: SemanticAction, world: "WorldState") -> ValidationResult:
    return _ripple_mood(action, world, EmotionalState.FRIENDLY, radius=2)


def _compile_cough(action: SemanticAction, world: "WorldState") -> ValidationResult:
    """Cough — airborne droplets at actor tile; sick actors may spread influenza."""
    actor = _actor(action, world)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)
    from .contamination import cough_aerosol_transitions

    pathogen = "cough_droplets"
    inf = actor.meta.get("infection")
    if isinstance(inf, dict) and inf.get("stage") == "sick":
        agent = str(inf.get("agent", ""))
        if agent in ("influenza", "cough_droplets"):
            pathogen = "influenza" if agent == "influenza" else "cough_droplets"
    transitions = cough_aerosol_transitions(world, actor, pathogen=pathogen)
    transitions.append(Transition(
        kind=TransitionKind.NOISE_EVENT,
        payload={
            "source_id": str(actor.entity_id),
            "source_pos": {"x": actor.position.x, "y": actor.position.y},
            "noise_level": 25,
        },
    ))
    return ValidationResult(valid=True, concrete_transitions=transitions, plausibility=1.0)


def _compile_sigh(action: SemanticAction, world: "WorldState") -> ValidationResult:
    actor = _actor(action, world)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)
    transitions = [Transition(
        kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED,
        payload={"entity_id": str(actor.entity_id), "to": EmotionalState.NEUTRAL.value},
    )]
    return ValidationResult(valid=True, concrete_transitions=transitions, plausibility=1.0)


def _ripple_mood(
    action: SemanticAction,
    world: "WorldState",
    to_state: EmotionalState,
    *,
    radius: int,
) -> ValidationResult:
    actor = _actor(action, world)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)
    transitions = _mood_ripple_transitions(actor, world, to_state, radius=radius)
    return ValidationResult(valid=True, concrete_transitions=transitions, plausibility=1.0)


def _mood_ripple_transitions(
    actor: EntityState,
    world: "WorldState",
    to_state: EmotionalState,
    *,
    radius: int,
) -> list[Transition]:
    transitions: list[Transition] = []
    for ent in _nearby_entities(world, actor.position, radius=radius, exclude=actor.entity_id):
        # Don't override deep hostility with laughter.
        if ent.emotional_state in (EmotionalState.HOSTILE, EmotionalState.ANGRY):
            continue
        transitions.append(Transition(
            kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED,
            payload={"entity_id": str(ent.entity_id), "to": to_state.value},
        ))
    return transitions


# ── Contact verbs ────────────────────────────────────────────────────────


def _compile_contact_verb(
    action: SemanticAction,
    world: "WorldState",
    *,
    aggression: int,
    mood_to: Optional[EmotionalState],
    damage: int = 0,
    wet: bool = False,
    edge_delta: float = 0.0,
    edge_kind: EdgeKind = EdgeKind.INTERACTED,
) -> ValidationResult:
    actor = _actor(action, world)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)
    if not action.target:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.MALFORMED_ACTION,
            rejection_detail="Contact verb needs a target.",
        )
    target = _resolve_entity(world, str(action.target))
    if target is None:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.TARGET_NOT_FOUND,
        )
    if actor.position.manhattan(target.position) > 1:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=f"Too far to {action.verb} {target.name}.",
        )

    transitions: list[Transition] = [
        Transition(
            kind=TransitionKind.CONTACT_INITIATED,
            payload={
                "actor": str(actor.entity_id),
                "target": str(target.entity_id),
                "manner": str(action.verb),
                "aggression": aggression,
                "consent_state": "unwelcomed" if aggression > 50 else "neutral",
                "surprise": target.alertness == AlertnessLevel.UNAWARE,
            },
        ),
    ]
    if mood_to is not None:
        transitions.append(Transition(
            kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED,
            payload={"entity_id": str(target.entity_id), "to": mood_to.value},
        ))
    if damage > 0:
        transitions.append(Transition(
            kind=TransitionKind.ENTITY_HEALTH_CHANGED,
            payload={
                "entity_id": str(target.entity_id),
                "delta": -damage,
                "cause": str(action.verb),
                "actor": str(actor.entity_id),
            },
        ))
    if wet:
        transitions.append(Transition(
            kind=TransitionKind.FLUID_CHANGED,
            payload={
                "target_kind": "entity",
                "entity_id": str(target.entity_id),
                "material": "saliva",
                "volume_ml": 5.0,
                "cause": str(action.verb),
                "wet": True,
            },
        ))
    if edge_delta != 0.0:
        transitions.append(Transition(
            kind=TransitionKind.EDGE_UPDATED,
            payload={
                "source": str(target.entity_id),
                "target": str(actor.entity_id),
                "edge_kind": edge_kind.value,
                "delta": edge_delta,
                "meta": {"verb": str(action.verb)},
            },
        ))
    return ValidationResult(valid=True, concrete_transitions=transitions, plausibility=1.0)


def _compile_pat(a, w):
    return _compile_contact_verb(a, w, aggression=5, mood_to=EmotionalState.FRIENDLY,
                                  edge_delta=0.05, edge_kind=EdgeKind.RESPECTS)


def _compile_hug(a, w):
    return _compile_contact_verb(a, w, aggression=0, mood_to=EmotionalState.HAPPY,
                                  edge_delta=0.1, edge_kind=EdgeKind.ALLY_OF)


def _compile_kiss(a, w):
    return _compile_contact_verb(a, w, aggression=0, mood_to=EmotionalState.HAPPY,
                                  edge_delta=0.15, edge_kind=EdgeKind.ALLY_OF)


def _compile_slap(a, w):
    return _compile_contact_verb(a, w, aggression=70, mood_to=EmotionalState.ANGRY,
                                  damage=2, edge_delta=0.2, edge_kind=EdgeKind.ENEMY_OF)


def _compile_spit(a, w):
    return _compile_contact_verb(a, w, aggression=80, mood_to=EmotionalState.ANGRY,
                                  wet=True, edge_delta=0.25, edge_kind=EdgeKind.ENEMY_OF)


def _compile_shove(a, w):
    return _compile_contact_verb(a, w, aggression=65, mood_to=EmotionalState.ANGRY,
                                  damage=1, edge_delta=0.15, edge_kind=EdgeKind.ENEMY_OF)


def _compile_nudge(a, w):
    return _compile_contact_verb(a, w, aggression=15, mood_to=None,
                                  edge_delta=0.02, edge_kind=EdgeKind.INTERACTED)


# ── Registry ─────────────────────────────────────────────────────────────


INTERACTION_COMPILERS: dict[str, callable] = {
    # Posture
    "sit":      _compile_sit,
    "stand":    _compile_stand,
    "lean":     _compile_lean,
    "slouch":   _compile_lean,
    "kneel":    _compile_kneel,
    "bow":      _compile_kneel,  # bow → kneeling-like reverence
    # Hygiene
    "wipe":     _compile_wipe,
    "clean":    _compile_clean,
    "polish":   _compile_clean,
    "relieve":  _compile_relieve,
    "wash":     _compile_wipe,
    # Sustenance
    "eat":      _compile_eat,
    "nibble":   _compile_eat,
    "munch":    _compile_eat,
    # Perception
    "sniff":    _compile_sniff,
    "smell":    _compile_sniff,
    "listen":   _compile_listen,
    # Expressive
    "laugh":    _compile_laugh,
    "chuckle":  _compile_laugh,
    "sing":     _compile_sing,
    "hum":      _compile_hum,
    "sigh":     _compile_sigh,
    "cough":    _compile_cough,
    "sneeze":   _compile_cough,
    # Contact
    "pat":      _compile_pat,
    "hug":      _compile_hug,
    "embrace":  _compile_hug,
    "kiss":     _compile_kiss,
    "slap":     _compile_slap,
    "spit":     _compile_spit,
    "shove":    _compile_shove,
    "push":     _compile_shove,
    "nudge":    _compile_nudge,
    "elbow":    _compile_nudge,
}
