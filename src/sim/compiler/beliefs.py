"""Belief propagation transitions."""

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
)
from ..spatial import (
    entities_visible_from,
    find_path,
    is_reachable,
    visible_from,
)
from .common import *

def build_belief_transitions(
    action: SemanticAction,
    grounding: "GroundingResult",
    validation: ValidationResult,
    world: WorldState,
) -> list[Transition]:
    """
    Zone 2 (EXTENDING): after a social or world-authoring action succeeds,
    write persistent state changes that represent player-asserted facts
    becoming canonical world state.

    Four classes of EXTENDING action are handled:

      1. Social claim     — ``asserted_fact`` is a plain claim about a person
                            or relationship → BELIEVES_CLAIM edge on target.

      2. Faction founding — ``asserted_fact`` starts with "found:" or "create
                            faction:" → FACTION_CREATED + RELATIONAL_NODE_CREATED,
                            actor and any named members join it.

      3. Relationship     — ``asserted_fact`` starts with "relationship:" →
                            EDGE_CREATED between named entities with the stated
                            edge kind (e.g. "relationship:owe_debt:guard:merchant").

      4. World fact       — ``asserted_fact`` starts with "world:" → a
                            RELATIONAL_NODE_CREATED representing a new concept,
                            location, or organisation the player is asserting.

    All EXTENDING writes live in the relational graph as uncertain/revocable
    structures (belief edges, faction nodes) — not as ground-truth changes to
    EntityState scalars.  The player can assert, but the simulation decides
    downstream consequences.
    """
    if not validation.valid:
        return []
    if grounding.zone != ActionZone.EXTENDING:
        return []
    if not grounding.asserted_fact:
        return []

    transitions: list[Transition] = []
    fact = grounding.asserted_fact.strip()
    actor_id = str(action.actor)
    import uuid as _uuid

    # ── Faction founding ──────────────────────────────────────────────────
    faction_prefixes = ("found:", "create faction:", "establish faction:", "form faction:")
    for prefix in faction_prefixes:
        if fact.lower().startswith(prefix):
            faction_name = fact[len(prefix):].strip().title()
            faction_id = f"faction_{faction_name.lower().replace(' ', '_')}_{_uuid.uuid4().hex[:6]}"
            # Actor and any explicitly named members in asserted_fact
            members = [actor_id]
            if action.target and not isinstance(action.target, type(None)):
                members.append(str(action.target))
            transitions.append(Transition(
                kind=TransitionKind.FACTION_CREATED,
                payload={
                    "faction_id": faction_id,
                    "name": faction_name,
                    "description": f"Founded by player decree: '{fact}'",
                    "founding_entity_id": actor_id,
                    "initial_members": members,
                },
            ))
            return transitions

    # ── Explicit relationship claim ───────────────────────────────────────
    # Format: "relationship:<edge_kind>:<entity_a>:<entity_b>"
    if fact.lower().startswith("relationship:"):
        parts = fact.split(":", 3)
        if len(parts) >= 4:
            _, edge_kind_str, ent_a_name, ent_b_name = parts
            grid = world.spatial
            ent_a = _resolve_entity_target(grid, ent_a_name.strip())
            ent_b = _resolve_entity_target(grid, ent_b_name.strip())
            if ent_a and ent_b:
                # Map string to EdgeKind — allow unknown kinds via a fallback
                try:
                    ek = EdgeKind(edge_kind_str.strip().lower())
                except ValueError:
                    ek = EdgeKind.INTERACTED  # fallback
                transitions.append(Transition(
                    kind=TransitionKind.EDGE_CREATED,
                    payload={
                        "source": str(ent_a.entity_id),
                        "target": str(ent_b.entity_id),
                        "edge_kind": ek.value,
                        "weight": validation.plausibility * 0.8,
                        "meta": {"claim": fact, "asserted_by": actor_id},
                    },
                ))
                return transitions

    # ── World fact / new node ─────────────────────────────────────────────
    world_prefixes = ("world:", "place:", "organisation:", "organization:", "group:")
    for prefix in world_prefixes:
        if fact.lower().startswith(prefix):
            label = fact[len(prefix):].strip()
            node_kind = prefix.rstrip(":").lower()
            if node_kind in ("organisation", "organization"):
                node_kind = "organization"
            node_id = f"node_{label.lower().replace(' ', '_')}_{_uuid.uuid4().hex[:6]}"
            transitions.append(Transition(
                kind=TransitionKind.RELATIONAL_NODE_CREATED,
                payload={
                    "node_id": node_id,
                    "node_kind": node_kind,
                    "label": label,
                    "meta": {"asserted_by": actor_id, "claim": fact},
                    "founding_entity_id": actor_id,
                },
            ))
            return transitions

    # ── Default: social claim propagated via BELIEVES_CLAIM edge ─────────
    if action.target is None or isinstance(action.target, type(None)):
        return []

    target_id = str(action.target)

    # Only write the belief edge when the social action actually succeeded
    succeeded = any(
        t.kind == TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED
        and t.payload.get("to") in (
            EmotionalState.FRIENDLY.value,
            EmotionalState.FEARFUL.value,
            EmotionalState.NEUTRAL.value,
        )
        for t in validation.concrete_transitions
    )
    # Also accept any positive dialogue exchange as success signal
    if not succeeded:
        succeeded = any(
            t.kind == TransitionKind.DIALOGUE_SPOKEN
            for t in validation.concrete_transitions
        )
    if not succeeded:
        return []

    transitions.append(Transition(
        kind=TransitionKind.EDGE_CREATED,
        payload={
            "source": target_id,
            "target": actor_id,
            "edge_kind": EdgeKind.BELIEVES_CLAIM.value,
            "weight": validation.plausibility,
            "meta": {
                "claim": fact,
                "verb": action.verb,
            },
        },
    ))
    return transitions

