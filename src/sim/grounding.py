"""
Grounding classifier.

Runs before the Compiler and answers one question:
  "How well is this SemanticAction anchored in canonical world state?"

Returns an ActionZone:

  GROUNDED  (Zone 1)
    All referenced entities exist in the spatial grid or relational graph.
    The Compiler handles everything.

  EXTENDING (Zone 2)
    The player is asserting/inventing a world fact (a lie, a claim, an
    assumed identity) that does not yet exist in canonical state but
    could plausibly attach to it.  The Compiler resolves the social
    interaction and additionally writes a BELIEVES_CLAIM edge recording
    what the target now believes.

  ORPHANED  (Zone 3)
    The action references nothing canonical and cannot extend the model
    because there is no scaffolding for it in the world.  The Compiler
    records a trivial WAIT-like event and the NarratorActor provides
    the only meaningful response.

This module is PURELY READ-ONLY.  It never mutates world state.
It makes no LM calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .schemas import (
    ActionType,
    ActionZone,
    Coord,
    EntityId,
    GroundingResult,
    SemanticAction,
    WorldState,
)

# English stop words to strip when synthesizing a verb from raw intent.
_STOP_WORDS = frozenset({
    "a", "an", "the", "with", "to", "at", "on", "in", "of", "for",
    "and", "or", "by", "from", "up", "out", "my", "your", "his", "her",
    "its", "our", "their", "guard", "player", "npc",
})


def _intent_to_verb(raw_intent: str) -> str:
    """
    Synthesize a freeform verb string from a raw player intent.

    Strategy: lowercase, split on whitespace/punctuation, drop stop words
    and known NPC names, join the first 1-3 meaningful tokens with
    underscores.

    Examples:
      "Shake Hands with guard"  → "shake_hands"
      "Give guard a hug"        → "give_hug"
      "Bow before the merchant" → "bow"
      "High five Mira"          → "high_five"
    """
    import re
    tokens = re.split(r"[\s\-_']+", raw_intent.lower())
    meaningful = [
        t for t in tokens
        if t and t not in _STOP_WORDS and t.isalpha()
    ]
    verb = "_".join(meaningful[:3]) if meaningful else "interact"
    return verb or "interact"


# ── Action types that can assert invented facts (Zone 2 candidates) ──────────
_ASSERTING_TYPES = {
    ActionType.DECEIVE,
    ActionType.PERSUADE,
    ActionType.SPEAK,
    ActionType.ASK,
    ActionType.BRIBE,
    ActionType.THREATEN,
}

# ── Literal strings that mean the player genuinely wanted to wait ─────────────
_TRIVIAL_WAIT_INTENTS = {"wait", "w", "pass", "skip", "do nothing", ""}

# ── Action types that are always symbolic unless the world has affordances ────
_ALWAYS_SYMBOLIC = {ActionType.SYMBOLIC}

# ── Orientation verbs: their "target" is a direction string, not an entity id.
# Always treat as Zone 1 (GROUNDED) — the compiler validates direction strings.
_ORIENTATION_VERBS = {ActionType.TURN, ActionType.LOOK, ActionType.FACE}

# Note: there is no module-level affordance map. Affordance rules are
# loaded per-world from affordances.yaml and live on world.config. The
# engine has no opinion about prayer, magic, or any other thematic
# domain — every world declares its own vocabulary.


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def classify(
    action: SemanticAction,
    world: WorldState,
    original_intent: str = "",
) -> GroundingResult:
    """
    Classify a SemanticAction into Zone 1, 2, or 3.

    Parameters
    ----------
    action          : The action to classify (post-LM, pre-Compiler).
    world           : Canonical world state (read-only).
    original_intent : The raw player text that produced the action.
                      Used to detect orphaned WAITs.
    """
    grid = world.spatial
    graph = world.relational

    # ── Actor must always exist ───────────────────────────────────────────────
    actor_exists = action.actor in grid.entities
    if not actor_exists:
        return GroundingResult(
            zone=ActionZone.ORPHANED,
            orphan_reason=f"Actor '{action.actor}' not found in world.",
        )

    # ── Resolve target ────────────────────────────────────────────────────────
    target_entity_id: Optional[EntityId] = None
    target_is_coord = isinstance(action.target, Coord)
    if action.target is not None and not target_is_coord:
        target_entity_id = EntityId(str(action.target))

    target_exists = (
        action.target is None
        or target_is_coord
        or (target_entity_id is not None and target_entity_id in grid.entities)
    )

    grounded_entities: list[EntityId] = [action.actor]
    if target_entity_id and target_entity_id in grid.entities:
        grounded_entities.append(target_entity_id)

    # ── Orientation verbs: directional targets are never entity IDs ──────────
    # "turn north", "face left", "look east" → always Zone 1.  The compiler
    # owns the validation logic for these; the grounding layer has no opinion.
    if action.verb in _ORIENTATION_VERBS:
        return GroundingResult(
            zone=ActionZone.GROUNDED,
            grounded_entities=[action.actor],
        )

    # ── Zone 3 check A: always-symbolic action types ──────────────────────────
    if action.verb in _ALWAYS_SYMBOLIC:
        affordance_match = _check_world_affordances(action, original_intent, world)
        if affordance_match:
            # World has scaffolding — treat as GROUNDED
            return GroundingResult(
                zone=ActionZone.GROUNDED,
                grounded_entities=grounded_entities,
            )
        return GroundingResult(
            zone=ActionZone.ORPHANED,
            grounded_entities=grounded_entities,
            orphan_reason=_orphan_reason_for(action, original_intent, world),
        )

    # ── Zone 3 check B: WAIT that came from a non-trivial intent ─────────────
    normalized_intent = original_intent.lower().strip()
    if action.verb == ActionType.WAIT:
        if normalized_intent not in _TRIVIAL_WAIT_INTENTS:
            # Interaction rescue: the LM correctly identified a target entity
            # but failed to coin a verb and defaulted to WAIT.  If a visible
            # entity target exists, synthesize a verb from the raw intent and
            # treat the action as Zone 1 (GROUNDED freeform).  This lets small
            # models handle "shake hands with guard", "bow to the noble", etc.
            # without needing to enumerate every possible social verb.
            if (
                target_entity_id is not None
                and target_entity_id in grid.entities
            ):
                synthesized = _intent_to_verb(original_intent)
                # Return a grounded result; the game_loop will need to re-run
                # compile_action on the synthesized verb.  We stash it in
                # orphan_reason so the caller can retrieve it — the grounding
                # contract lets orphan_reason carry diagnostic text even on
                # non-ORPHANED results (it is ignored by the compiler when
                # zone != ORPHANED).
                return GroundingResult(
                    zone=ActionZone.GROUNDED,
                    grounded_entities=grounded_entities,
                    orphan_reason=synthesized,  # stash synthesized verb here
                )

            # The LM couldn't map this intent to anything physical.
            # Check if any affordance keywords appear in the intent.
            if _check_world_affordances(action, original_intent, world):
                return GroundingResult(
                    zone=ActionZone.GROUNDED,
                    grounded_entities=grounded_entities,
                )
            return GroundingResult(
                zone=ActionZone.ORPHANED,
                grounded_entities=grounded_entities,
                orphan_reason=_orphan_reason_for(action, original_intent, world),
            )
        # Genuine WAIT intent — Zone 1
        return GroundingResult(
            zone=ActionZone.GROUNDED,
            grounded_entities=grounded_entities,
        )

    # ── Zone 3 check C: target doesn't exist ─────────────────────────────────
    if not target_exists and target_entity_id is not None:
        # The action names an entity that's not in the world.
        # For asserting action types, this is Zone 2 (player invents a person).
        # For all others it's a malformed/orphaned action.
        if action.verb in _ASSERTING_TYPES and action.intent.rationale:
            return GroundingResult(
                zone=ActionZone.EXTENDING,
                grounded_entities=[action.actor],
                asserted_fact=_extract_asserted_fact(action),
            )
        return GroundingResult(
            zone=ActionZone.ORPHANED,
            grounded_entities=[action.actor],
            orphan_reason=(
                f"Target entity '{target_entity_id}' does not exist "
                f"and action type '{action.verb}' cannot extend the world."
            ),
        )

    # ── Zone 2 check: asserting action with a non-trivial rationale ──────────
    if (
        action.verb in _ASSERTING_TYPES
        and target_exists
        and action.intent.rationale
        and _looks_like_assertion(action.intent.rationale)
    ):
        return GroundingResult(
            zone=ActionZone.EXTENDING,
            grounded_entities=grounded_entities,
            asserted_fact=_extract_asserted_fact(action),
        )

    # ── Default: Zone 1 ───────────────────────────────────────────────────────
    return GroundingResult(
        zone=ActionZone.GROUNDED,
        grounded_entities=grounded_entities,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _check_world_affordances(
    action: SemanticAction,
    original_intent: str,
    world: WorldState,
) -> bool:
    """
    Return True if the world's relational graph contains a node that
    satisfies any affordance rule whose keyword appears in the intent.

    Rules come from world.config.affordances (loaded from the world
    pack's affordances.yaml). A node satisfies a rule when:

        node.kind in rule.kinds
        AND ( any name_keyword appears in node.name.lower()
              OR any rule.tag appears in node.meta.get("tags", []) )

    If the world has no affordance rules at all, the answer is False —
    nothing is anchored, and any symbolic action falls to Zone 3.
    """
    rules = world.config.affordances
    if not rules:
        return False
    text = (original_intent + " " + (action.intent.rationale or "")).lower()
    graph = world.relational
    for rule in rules:
        if rule.keyword not in text:
            continue
        allowed_kinds = set(rule.kinds)
        name_keywords = [k.lower() for k in rule.name_keywords]
        rule_tags = set(rule.tags)
        for node in graph.nodes.values():
            if allowed_kinds and node.kind not in allowed_kinds:
                continue
            node_name_lower = node.name.lower()
            if any(nk in node_name_lower for nk in name_keywords):
                return True
            node_tags = set(node.meta.get("tags") or [])
            if rule_tags and node_tags & rule_tags:
                return True
    return False


def _orphan_reason_for(
    action: SemanticAction,
    original_intent: str,
    world: WorldState,
) -> str:
    """
    Produce a terse, generic explanation of why the action is orphaned.

    The reason is derived entirely from the world pack's affordance
    rules: if the intent mentions a keyword the pack declares (e.g. a
    pack that defines a "prayer" rule), but no node in the graph
    satisfies it, we say so by keyword. No hardcoded thematic
    categories live here.
    """
    intent_text = original_intent.strip() or action.verb
    lower = intent_text.lower()

    # Find affordance keywords mentioned in the intent — these are the
    # domains the player invoked that the world does not support.
    missing_keywords: list[str] = []
    for rule in world.config.affordances:
        if rule.keyword in lower:
            missing_keywords.append(rule.keyword)

    if missing_keywords:
        kws = ", ".join(repr(k) for k in missing_keywords)
        return (
            f"The world declares affordance rules for {kws}, but no node "
            f"in the relational graph currently satisfies them."
        )

    return (
        f"Intent '{intent_text[:60]}' references no entity, object, or "
        f"faction currently in the world."
    )


def _looks_like_assertion(rationale: str) -> bool:
    """
    Heuristic: does the rationale look like the player is claiming something
    rather than just describing an emotional posture?
    """
    assertion_markers = [
        "i am", "i'm", "my name", "i work", "i know",
        "i have", "i was", "my father", "my brother", "my sister",
        "i represent", "sent by", "authorized", "cousin", "nephew",
        "friend of", "hired by", "permission", "pass", "key",
    ]
    lower = rationale.lower()
    return any(marker in lower for marker in assertion_markers)


def _extract_asserted_fact(action: SemanticAction) -> str:
    """Extract the claim the player is making as a plain string."""
    if action.intent.rationale:
        return action.intent.rationale[:200]
    if action.intent.desired_outcome:
        return "; ".join(action.intent.desired_outcome[:3])
    return f"{action.verb} by {action.actor}"
