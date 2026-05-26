"""
Property-intersection resolver.

Evaluates ``property_interactions.yaml`` rules inside the compiler AFTER
pack-authored content (verb templates + open verbs), but BEFORE the
engine's generic per-verb built-ins and the freeform LM fallback.

Matching is driven entirely by entity/object **tags** (what things physically
ARE) rather than by verb keywords (what the player typed).  This lets the
engine deterministically resolve novel actions like:

  "I pour my flask of acid on the iron door"
  → actor holds [corrosive] object, target has [metal] tag → rule fires

without the LM adjudicator needing to guess the outcome.

Resolution order within the compiler (``verb_pipeline.compile_unresolved_verb``):
  1.  Pack verb_templates                       (explicit author intent)
  2.  Pack open_verbs                           (explicit author intent)
  3.  **Property-intersection (this module)**   (deterministic physics — tag × tag)
  4.  Engine built-in defaults                  (generic per-verb fallback)
  5.  Contact-category reroute
  6.  Freeform fallback (LM proposals + contest)

Physics beats generic verb shortcuts so that creative material/object
combinations produce specific mechanical outcomes (corrosive on metal,
fire-source on flammable, leverage on locked) instead of degrading to a
generic OPEN / INTIMIDATE / FLEE handler.

Intent categories
-----------------
Each rule declares which action *categories* it fires on.  The category is
derived from the verb string, not from free text, so it is cheap and stable.

  any         – fires on all non-trivially-blocked categories (default)
  physical    – attack, strike, hit, cut, throw, smash, …
  chemical    – pour, apply, use, mix, combine, smear, coat, dip, …
  thermal     – heat, cool, ignite, melt, freeze, extinguish, …
  containment – fill, empty, store, retrieve, open, close, …
  movement    – move, run, walk, flee, sneak, …   (rarely useful for interactions)
  social      – speak, ask, persuade, threaten, …
  observation – look, examine, inspect, search, …
  idle        – wait, rest, sleep, …

Rules with ``intent_categories: [any]`` fire on all categories except social
and observation (which almost never produce physical contact).  If you want
a rule that fires even during speech — e.g. "your silver amulet passively
harms nearby undead" — add ``social`` explicitly.
"""

from __future__ import annotations

import logging
from typing import Optional

from .schemas import (
    EntityId,
    EntityState,
    IntentBlock,
    ObjectState,
    PropertyInteractionRule,
    RejectionReason,
    SemanticAction,
    StyleBlock,
    Transition,
    TransitionKind,
    TransitionProposal,
    ValidationResult,
    WorldState,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent-category classification
# ---------------------------------------------------------------------------

_PHYSICAL_VERBS: frozenset[str] = frozenset({
    "attack", "strike", "hit", "cut", "slash", "stab", "smash", "break",
    "shatter", "crush", "throw", "fling", "hurl", "push", "shove", "pull",
    "drag", "slam", "pound", "bash", "kick", "punch", "wrestle", "tear",
    "rip", "chop", "swing", "pry", "dislodge",
})

_CHEMICAL_VERBS: frozenset[str] = frozenset({
    "pour", "apply", "use", "mix", "combine", "smear", "coat", "dip",
    "splash", "spray", "drench", "anoint", "rub", "paint", "soak",
    "dissolve", "dilute", "fill", "empty", "transfer",
})

_THERMAL_VERBS: frozenset[str] = frozenset({
    "heat", "cool", "ignite", "light", "kindle", "torch", "burn",
    "melt", "freeze", "freeze", "extinguish", "douse", "quench",
    "roast", "boil", "sear", "char", "incinerate", "warm",
})

_CONTAINMENT_VERBS: frozenset[str] = frozenset({
    "fill", "empty", "store", "retrieve", "open", "close", "lock",
    "unlock", "seal", "unseal", "cork", "uncork",
})

_MOVEMENT_VERBS: frozenset[str] = frozenset({
    "move", "run", "walk", "flee", "sneak", "jump", "climb", "swim",
    "crawl", "roll", "dash", "sprint",
})

_SOCIAL_VERBS: frozenset[str] = frozenset({
    "speak", "say", "tell", "ask", "talk", "chat", "shout", "whisper",
    "call", "yell", "persuade", "convince", "threaten", "deceive",
    "lie", "bribe", "charm", "flatter", "insult", "taunt", "plead",
    "negotiate", "bargain",
})

_OBSERVATION_VERBS: frozenset[str] = frozenset({
    "look", "examine", "inspect", "search", "read", "listen", "watch",
    "observe", "scan", "check", "study", "peer", "squint",
})

_IDLE_VERBS: frozenset[str] = frozenset({
    "wait", "rest", "sleep", "meditate", "sit", "stand", "pause",
    "stay", "linger",
})

# Categories that "any" intentionally excludes.  Physical contact is
# essentially never initiated by speech or pure observation.
_ANY_EXCLUDES: frozenset[str] = frozenset({"social", "observation", "idle"})


def _intent_category(verb: str) -> str:
    v = verb.lower().strip()
    if v in _SOCIAL_VERBS:
        return "social"
    if v in _OBSERVATION_VERBS:
        return "observation"
    if v in _IDLE_VERBS:
        return "idle"
    if v in _MOVEMENT_VERBS:
        return "movement"
    if v in _THERMAL_VERBS:
        return "thermal"
    if v in _CHEMICAL_VERBS:
        return "chemical"
    if v in _CONTAINMENT_VERBS:
        return "containment"
    if v in _PHYSICAL_VERBS:
        return "physical"
    # Default: treat unknown verbs as physical actions.
    return "physical"


def _category_matches(rule: PropertyInteractionRule, verb: str) -> bool:
    cats = {c.lower() for c in rule.intent_categories}
    intent_cat = _intent_category(verb)
    # If the intent category is explicitly in the list, always match.
    if intent_cat in cats:
        return True
    # "any" catches all categories except the default-excluded ones,
    # but only when the category wasn't explicitly listed above.
    if "any" in cats:
        return intent_cat not in _ANY_EXCLUDES
    return False


# ---------------------------------------------------------------------------
# Tag helpers
# ---------------------------------------------------------------------------

def _entity_tags(entity: EntityState) -> frozenset[str]:
    base = frozenset(t.lower() for t in (entity.tags or []))
    # Also include material tag if set in meta.
    mat = (entity.meta or {}).get("material", "")
    if mat:
        return base | {mat.lower()}
    return base


def _object_tags(obj: ObjectState) -> frozenset[str]:
    base = frozenset(t.lower() for t in (obj.tags or []))
    mat = (obj.meta or {}).get("material", "")
    if mat:
        return base | {mat.lower()}
    return base


def _equipped_objects(actor: EntityState, world: WorldState) -> list[ObjectState]:
    """Return all objects currently equipped by the actor (weapon + armor + slots)."""
    grid = world.spatial
    oids: set[str] = set()
    if actor.equipped_weapon:
        oids.add(str(actor.equipped_weapon))
    if actor.equipped_armor:
        oids.add(str(actor.equipped_armor))
    for oid in (actor.equipped_slots or {}).values():
        oids.add(str(oid))
    out: list[ObjectState] = []
    for oid in oids:
        obj = grid.objects.get(oid)
        if obj is not None:
            out.append(obj)
    return out


def _actor_source_tags(actor: EntityState, world: WorldState) -> frozenset[str]:
    """Tags of the actor plus all equipped items — the 'source' side of interaction."""
    tags = _entity_tags(actor)
    for obj in _equipped_objects(actor, world):
        tags = tags | _object_tags(obj)
    return tags


def _has_any(need: frozenset[str], have: frozenset[str]) -> bool:
    """True when need is empty (no requirement) OR any need tag is in have."""
    return (not need) or bool(need & have)


# ---------------------------------------------------------------------------
# Payload substitution
# ---------------------------------------------------------------------------

def _resolve_target_entity(
    action: SemanticAction,
    world: WorldState,
) -> Optional[EntityState]:
    if action.target is None:
        return None
    from .schemas import Coord
    if isinstance(action.target, Coord):
        # Target is a tile — find first entity there.
        for ent in world.spatial.entities.values():
            if ent.position == action.target and ent.entity_id != action.actor:
                return ent
        return None
    tid = str(action.target)
    return world.spatial.entities.get(tid)


def _resolve_target_object(
    action: SemanticAction,
    world: WorldState,
) -> Optional[ObjectState]:
    if action.target is None:
        return None
    from .schemas import Coord
    if isinstance(action.target, Coord):
        return None
    tid = str(action.target)
    return world.spatial.objects.get(tid)


def _target_tags(
    target_ent: Optional[EntityState],
    target_obj: Optional[ObjectState],
) -> frozenset[str]:
    tags: frozenset[str] = frozenset()
    if target_ent is not None:
        tags = tags | _entity_tags(target_ent)
    if target_obj is not None:
        tags = tags | _object_tags(target_obj)
    return tags


def _target_position(
    action: SemanticAction,
    target_ent: Optional[EntityState],
    target_obj: Optional[ObjectState],
) -> Optional[tuple[int, int]]:
    from .schemas import Coord
    if isinstance(action.target, Coord):
        return (action.target.x, action.target.y)
    if target_ent is not None and target_ent.position is not None:
        return (target_ent.position.x, target_ent.position.y)
    if target_obj is not None and target_obj.position is not None:
        return (target_obj.position.x, target_obj.position.y)
    return None


def _substitute_payload(
    raw_payload: dict,
    *,
    actor_id: str,
    target_id: Optional[str],
    tool_id: str,
    actor_x: int,
    actor_y: int,
    target_x: Optional[int],
    target_y: Optional[int],
) -> dict:
    out: dict = {}
    for k, v in raw_payload.items():
        if v == "$actor":
            out[k] = actor_id
        elif v == "$target":
            out[k] = target_id or ""
        elif v == "$tool":
            out[k] = tool_id
        elif v == "$actor_x":
            out[k] = actor_x
        elif v == "$actor_y":
            out[k] = actor_y
        elif v == "$target_x":
            out[k] = target_x if target_x is not None else actor_x
        elif v == "$target_y":
            out[k] = target_y if target_y is not None else actor_y
        else:
            out[k] = v
    return out


def _materialise_effects(
    rule: PropertyInteractionRule,
    *,
    actor_id: str,
    target_id: Optional[str],
    tool_id: str,
    actor_x: int,
    actor_y: int,
    target_x: Optional[int],
    target_y: Optional[int],
) -> list[Transition]:
    transitions: list[Transition] = []
    for proposal in rule.effects:
        payload = _substitute_payload(
            dict(proposal.payload or {}),
            actor_id=actor_id,
            target_id=target_id,
            tool_id=tool_id,
            actor_x=actor_x,
            actor_y=actor_y,
            target_x=target_x,
            target_y=target_y,
        )
        try:
            kind = TransitionKind(proposal.kind)
        except ValueError:
            logger.warning(
                "PropertyInteractionRule %r: unknown TransitionKind %r — skipped",
                rule.id,
                proposal.kind,
            )
            continue
        transitions.append(Transition(kind=kind, payload=payload))
    return transitions


# ---------------------------------------------------------------------------
# Distance check
# ---------------------------------------------------------------------------

def _actor_target_distance(
    actor: EntityState,
    target_x: Optional[int],
    target_y: Optional[int],
) -> float:
    if target_x is None or target_y is None:
        return 0.0  # No position info → always within range.
    if actor.position is None:
        return 0.0
    dx = actor.position.x - target_x
    dy = actor.position.y - target_y
    return (dx * dx + dy * dy) ** 0.5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def try_property_interaction(
    action: SemanticAction,
    world: WorldState,
) -> Optional[ValidationResult]:
    """
    Attempt to resolve *action* via property-intersection rules.

    Returns a successful ``ValidationResult`` when a rule fires, or ``None``
    when no rule matches (letting the compiler continue to the next step).

    The highest-priority matching rule wins.  Ties are broken by rule order
    in the pack (first declared wins).
    """
    rules = world.config.property_interactions
    if not rules:
        return None

    actor = world.spatial.entities.get(action.actor)
    if actor is None:
        # Try region-aware lookup.
        try:
            grid = world.grid_for_entity(action.actor)
            actor = grid.entities.get(action.actor)
        except (KeyError, AttributeError):
            pass
    if actor is None:
        return None

    target_ent = _resolve_target_entity(action, world)
    target_obj = _resolve_target_object(action, world)

    # If the action has no target at all and all rules require a target, bail.
    if action.target is None and all(r.target_tags for r in rules):
        return None

    src_tags = _actor_source_tags(actor, world)
    tgt_tags = _target_tags(target_ent, target_obj)
    tgt_pos = _target_position(action, target_ent, target_obj)

    target_id: Optional[str] = None
    if target_ent is not None:
        target_id = str(target_ent.entity_id)
    elif target_obj is not None:
        target_id = str(target_obj.object_id)

    # Identify which (if any) equipped object is the source tool.
    tool_id = ""
    equipped = _equipped_objects(actor, world)
    src_tool_tags: frozenset[str] = frozenset()
    for obj in equipped:
        src_tool_tags = src_tool_tags | _object_tags(obj)
    # The "tool" for substitution is the highest-priority equipped item whose
    # tags overlap the matched rule's actor_or_tool_tags.

    best: Optional[PropertyInteractionRule] = None
    best_priority = -1

    for rule in rules:
        # 1. Intent category filter.
        if not _category_matches(rule, str(action.verb)):
            continue

        # 2. Actor-or-tool requirement.
        need_src = frozenset(t.lower() for t in rule.actor_or_tool_tags)
        if not _has_any(need_src, src_tags):
            continue

        # 3. Target requirement.
        need_tgt = frozenset(t.lower() for t in rule.target_tags)
        if not _has_any(need_tgt, tgt_tags):
            continue

        # 4. Immunity: suppress rule if target has any immunity tag.
        immunity = frozenset(t.lower() for t in rule.target_immunity_tags)
        if immunity and (immunity & tgt_tags):
            continue

        # 5. Distance.
        tgt_x, tgt_y = (tgt_pos or (None, None))
        dist = _actor_target_distance(actor, tgt_x, tgt_y)
        if dist > rule.max_distance:
            continue

        if rule.priority > best_priority:
            best = rule
            best_priority = rule.priority

    if best is None:
        return None

    # Identify tool object id for substitution.
    need_src = frozenset(t.lower() for t in best.actor_or_tool_tags)
    for obj in equipped:
        if need_src and (need_src & _object_tags(obj)):
            tool_id = str(obj.object_id)
            break

    actor_x = actor.position.x if actor.position else 0
    actor_y = actor.position.y if actor.position else 0
    tgt_x_val, tgt_y_val = tgt_pos or (None, None)

    transitions = _materialise_effects(
        best,
        actor_id=str(action.actor),
        target_id=target_id,
        tool_id=tool_id,
        actor_x=actor_x,
        actor_y=actor_y,
        target_x=tgt_x_val,
        target_y=tgt_y_val,
    )

    if not transitions:
        logger.debug(
            "PropertyInteractionRule %r matched but produced no valid transitions.",
            best.id,
        )
        return None

    # Handle contest if the rule declares one.
    plausibility = 1.0
    if best.contest:
        import random

        actor_val = int((actor.attributes or {}).get(best.contest.actor_attribute, 10))
        diff = int(best.contest.difficulty or 50)
        roll = random.randint(1, 20) + (actor_val - 10) // 2
        if roll < diff // 5:
            # Rolled very badly — full miss.
            return ValidationResult(
                valid=False,
                rejection_reason=RejectionReason.CONTEST_FAILED,
                rejection_detail=(
                    f"[{best.name}] {best.contest.actor_attribute} check failed "
                    f"(roll {roll} vs difficulty {diff})."
                ),
            )
        plausibility = min(1.0, roll / max(diff // 5, 1) * 0.5 + 0.5)

    if best.narrative:
        actor_name = actor.name
        target_name = (
            target_ent.name if target_ent else
            target_obj.name if target_obj else "it"
        )
        tool_name = ""
        for obj in equipped:
            if tool_id and str(obj.object_id) == tool_id:
                tool_name = obj.name
                break

        narrative_text = (
            best.narrative
            .replace("{actor_name}", actor_name)
            .replace("{target_name}", target_name)
            .replace("{tool_name}", tool_name)
        )
        transitions.append(Transition(
            kind=TransitionKind.DIALOGUE_SPOKEN,
            payload={
                "entity_id": str(action.actor),
                "line": narrative_text,
                "is_narration": True,
            },
        ))

    logger.debug(
        "PropertyInteractionRule %r resolved action %r (priority %d, %d transitions).",
        best.id,
        action.verb,
        best.priority,
        len(transitions),
    )

    return ValidationResult(
        valid=True,
        concrete_transitions=transitions,
        plausibility=plausibility,
    )


# ---------------------------------------------------------------------------
# Physical-possibility rescue (called from the adjudication cascade)
# ---------------------------------------------------------------------------

# How far to scan around the actor for candidate property-rule targets when
# rescuing an otherwise-orphaned action.  Bounded to keep the search cheap
# (O(rules * candidates) per failed action, only on the adjudication path).
_RESCUE_SCAN_RADIUS: int = 6

# When synthesizing a verb for a rescued action, we need a verb that the
# rule's intent_categories will accept.  Pick a canonical verb per category
# so the resolver's _category_matches gate fires cleanly.
_CATEGORY_TO_VERB: dict[str, str] = {
    "thermal": "ignite",
    "chemical": "apply",
    "physical": "strike",
    "containment": "open",
    "any": "use",
    # movement / social / observation / idle are intentionally absent —
    # we don't want to rescue an orphaned action by inventing a speech act
    # or a wait.  If a rule is *only* social/observation/idle, we skip it.
}


def _synthesize_verb_for_rule(rule: PropertyInteractionRule) -> Optional[str]:
    """
    Pick a verb whose ``_intent_category`` will satisfy ``rule.intent_categories``.

    Returns ``None`` when the rule only fires on categories that are not
    safe to invent (social/observation/idle) — in that case, rescuing the
    action would mean putting words in the player's mouth.
    """
    cats = [c.lower() for c in (rule.intent_categories or [])] or ["any"]
    # Prefer the most physically-specific category the rule accepts.
    preference = ["thermal", "chemical", "physical", "containment", "any"]
    for cat in preference:
        if cat in cats:
            return _CATEGORY_TO_VERB[cat]
    return None


def _scan_candidate_targets(
    actor: EntityState,
    world: WorldState,
    radius: int,
) -> list[tuple[str, frozenset[str], float, bool]]:
    """
    Return (target_id, tags, distance, is_entity) for every entity/object
    within ``radius`` tiles of the actor.  Skips the actor itself.
    """
    if actor.position is None:
        return []
    ax, ay = actor.position.x, actor.position.y
    out: list[tuple[str, frozenset[str], float, bool]] = []

    for eid, ent in world.spatial.entities.items():
        if eid == actor.entity_id or ent.position is None:
            continue
        dx = ent.position.x - ax
        dy = ent.position.y - ay
        dist = (dx * dx + dy * dy) ** 0.5
        if dist <= radius:
            tags = _entity_tags(ent)
            if tags:
                out.append((str(eid), tags, dist, True))

    for oid, obj in world.spatial.objects.items():
        if obj.position is None:
            continue
        dx = obj.position.x - ax
        dy = obj.position.y - ay
        dist = (dx * dx + dy * dy) ** 0.5
        if dist <= radius:
            tags = _object_tags(obj)
            if tags:
                out.append((str(oid), tags, dist, False))

    return out


def try_property_possibility_rescue(
    action: SemanticAction,
    world: WorldState,
    intent: str,
) -> Optional[tuple[SemanticAction, ValidationResult]]:
    """
    Last deterministic rescue before LM adjudication.

    When an action has failed to compile and would otherwise either get
    LM-adjudicated or fall through to Zone-3 atmospheric prose, scan the
    actor's available source tags (entity + equipped/inventory items)
    against every property-interaction rule and every nearby candidate
    target.  If a rule *could* fire — i.e. the actor genuinely has the
    physical means to act on something nearby — synthesize a real
    ``SemanticAction`` that the rule will accept and compile it.

    Returns
    -------
    (synthesized_action, validation_result) when a rescue fired,
    ``None`` when no physical possibility exists (action is genuinely Z3).

    Design notes
    ------------
    - The original intent text is preserved on the synthesized action so
      narration still reads as the player's words rather than the
      engine's invented verb.
    - Only rules whose ``intent_categories`` include a physical category
      (thermal / chemical / physical / containment / any) are eligible —
      we never invent social, observation, or idle acts on the player's
      behalf, because doing so would put words in their mouth.
    - The highest-priority rule wins.  Ties broken by shortest distance.
    """
    rules = world.config.property_interactions
    if not rules:
        return None

    actor = world.spatial.entities.get(action.actor)
    if actor is None:
        try:
            grid = world.grid_for_entity(action.actor)
            actor = grid.entities.get(action.actor)
        except (KeyError, AttributeError):
            pass
    if actor is None or actor.position is None:
        return None

    src_tags = _actor_source_tags(actor, world)
    if not src_tags:
        # Actor has no tags and no equipped/inventory tagged items —
        # nothing to physically lever on the world with.
        return None

    candidates = _scan_candidate_targets(actor, world, _RESCUE_SCAN_RADIUS)
    if not candidates:
        return None

    best: Optional[tuple[PropertyInteractionRule, str, bool, float]] = None
    best_key: tuple[int, float] = (-1, float("inf"))  # (priority desc, dist asc)

    for rule in rules:
        # Skip rules that have no physically-safe intent category.
        if _synthesize_verb_for_rule(rule) is None:
            continue

        need_src = frozenset(t.lower() for t in rule.actor_or_tool_tags)
        if not _has_any(need_src, src_tags):
            continue

        need_tgt = frozenset(t.lower() for t in rule.target_tags)
        immunity = frozenset(t.lower() for t in rule.target_immunity_tags)

        for tid, ttags, tdist, is_entity in candidates:
            if tdist > rule.max_distance:
                continue
            if not _has_any(need_tgt, ttags):
                continue
            if immunity and (immunity & ttags):
                continue
            key = (rule.priority, -tdist)  # higher priority wins; then closer
            if (key[0], -key[1]) > (best_key[0], best_key[1]):
                # equivalent to: priority > best_priority OR
                # (priority == best_priority AND tdist < best_dist)
                best = (rule, tid, is_entity, tdist)
                best_key = (rule.priority, tdist)

    if best is None:
        return None

    rule, target_id, _is_entity, _dist = best
    synth_verb = _synthesize_verb_for_rule(rule)
    if synth_verb is None:
        return None  # defensive; _synthesize was None-checked above

    # Preserve the original intent text so the narrator still uses the
    # player's words; only the verb/target change for the resolver.
    style = action.style or StyleBlock(aggression=30, visibility=50)
    intent_block = action.intent or IntentBlock(rationale=intent or "")
    rescued_action = action.model_copy(
        update={
            "verb": synth_verb,
            "target": target_id,
            "style": style,
            "intent": intent_block,
            # raw_input is preserved verbatim — narration reads as the
            # player's original phrasing, not the synthesized verb.
        }
    )

    rescued_result = try_property_interaction(rescued_action, world)
    if rescued_result is None or not rescued_result.valid:
        return None

    logger.info(
        "property-possibility rescue fired: rule=%r verb=%r target=%r "
        "(intent=%r)",
        rule.id,
        synth_verb,
        target_id,
        (intent or "")[:60],
    )
    return rescued_action, rescued_result
