"""
Interaction grammar resolver (Overhaul A).

Deterministic [tag] × [tag] matching from ``interaction_rules.yaml``.
Runs after kernel pre-parsers and before LM ``infer()``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .projection import visible_from
from .schemas import (
    Affordance,
    EntityId,
    EntityState,
    IntentBlock,
    InteractionRule,
    ObjectState,
    SemanticAction,
    StyleBlock,
    TransitionProposal,
    WorldState,
)

_USE_ON_RE = re.compile(
    r"\b(use|apply|pour|spill|tip|empty|light|ignite|extinguish|douse|smear|coat|mix|combine)\b"
    r".{0,40}\b(on|onto|into|with|at)\b",
    re.I,
)
_SUBSTANCE_WORDS = frozenset({
    "oil", "ale", "water", "wine", "torch", "fire", "flame", "poison", "acid",
    "drink", "liquid", "flask", "barrel", "cask", "hearth", "chair", "table",
})


@dataclass
class InteractionSlots:
    text: str
    lower: str
    verb_tokens: list[str]
    substances: list[str]
    target_name: Optional[str] = None
    tool_name: Optional[str] = None


def extract_interaction_slots(text: str) -> InteractionSlots:
    lower = text.lower().strip()
    tokens = re.split(r"[\s\-_']+", lower)
    verb_tokens = [t for t in tokens if t.isalpha() and len(t) > 2]
    substances = [t for t in tokens if t in _SUBSTANCE_WORDS]

    target_name = None
    m = re.search(
        r"\b(?:on|onto|into|at|toward|towards|with)\s+(?:the\s+)?([a-z][a-z\s]{1,30}?)"
        r"(?:\s+(?:and|then|before|after|to)\b|$|\.)",
        lower,
    )
    if m:
        target_name = m.group(1).strip()

    tool_name = None
    m2 = re.search(
        r"\b(?:use|with|using|from)\s+(?:the\s+|my\s+)?([a-z][a-z\s]{1,24}?)"
        r"\s+(?:on|onto|to|against)\b",
        lower,
    )
    if m2:
        tool_name = m2.group(1).strip()

    return InteractionSlots(
        text=text,
        lower=lower,
        verb_tokens=verb_tokens,
        substances=substances,
        target_name=target_name,
        tool_name=tool_name,
    )


def _tags(obj) -> set[str]:
    return {t.lower() for t in (getattr(obj, "tags", None) or [])}


def _has_any(need: list[str], have: set[str]) -> bool:
    if not need:
        return True
    need_l = {t.lower() for t in need}
    return bool(have.intersection(need_l))


def _resolve_visible_entity(
    world: WorldState,
    actor: EntityState,
    name_hint: Optional[str],
) -> Optional[tuple[EntityId, EntityState]]:
    if not name_hint:
        return None
    hint = name_hint.lower()
    visible = visible_from(world.spatial, actor.position, actor.sight_range)
    best: Optional[tuple[EntityId, EntityState, int]] = None
    for eid, ent in world.spatial.entities.items():
        if not ent.alive or ent.entity_id == actor.entity_id:
            continue
        if ent.position not in visible:
            continue
        name_l = ent.name.lower()
        if hint in name_l or any(w in name_l for w in hint.split() if len(w) > 2):
            dist = actor.position.manhattan(ent.position)
            if best is None or dist < best[2]:
                best = (eid, ent, dist)
    return (best[0], best[1]) if best else None


def _resolve_visible_object(
    world: WorldState,
    actor: EntityState,
    name_hint: Optional[str],
) -> Optional[ObjectState]:
    visible = visible_from(world.spatial, actor.position, actor.sight_range)
    candidates: list[tuple[ObjectState, int]] = []
    for obj in world.spatial.objects.values():
        if obj.position and obj.position not in visible:
            continue
        if obj.position and actor.position.manhattan(obj.position) > 12:
            continue
        candidates.append((obj, actor.position.manhattan(obj.position) if obj.position else 99))
    if name_hint:
        hint = name_hint.lower()
        named = [
            (o, d) for o, d in candidates
            if hint in o.name.lower() or any(w in o.name.lower() for w in hint.split() if len(w) > 2)
        ]
        if named:
            return min(named, key=lambda x: x[1])[0]
    if candidates:
        return min(candidates, key=lambda x: x[1])[0]
    return None


def _inventory_objects(actor: EntityState, world: WorldState) -> list[ObjectState]:
    out: list[ObjectState] = []
    for oid in actor.inventory:
        obj = world.spatial.objects.get(oid)
        if obj is not None:
            out.append(obj)
    return out


def _resolve_tool(
    actor: EntityState,
    world: WorldState,
    tool_tags: list[str],
    name_hint: Optional[str],
) -> Optional[ObjectState]:
    items = _inventory_objects(actor, world)
    if name_hint:
        hint = name_hint.lower()
        for obj in items:
            if hint in obj.name.lower():
                return obj
    if tool_tags:
        need = {t.lower() for t in tool_tags}
        for obj in items:
            if _tags(obj).intersection(need):
                return obj
    return None


def _score_rule(
    rule: InteractionRule,
    slots: InteractionSlots,
    actor: EntityState,
    target_ent: Optional[EntityState],
    target_obj: Optional[ObjectState],
    tool: Optional[ObjectState],
) -> float:
    score = float(rule.priority)
    if rule.verb_keywords:
        if not any(kw in slots.lower for kw in rule.verb_keywords):
            return -1.0
        score += 2.0
    if rule.substance_keywords:
        if not any(kw in slots.lower for kw in rule.substance_keywords):
            return -1.0
        score += 1.5
    if not _has_any(rule.actor_tags, _tags(actor)):
        return -1.0
    if rule.tool_tags and tool is None:
        return -1.0
    if rule.tool_tags and not _has_any(rule.tool_tags, _tags(tool)):
        return -1.0
    if rule.tool_tags and tool is not None:
        score += 1.0

    kind = rule.target_kind.lower()
    if kind == "entity":
        if target_ent is None:
            return -1.0
        if not _has_any(rule.target_tags, _tags(target_ent)):
            return -1.0
        score += 1.0
    elif kind == "object":
        if target_obj is None:
            return -1.0
        if not _has_any(rule.target_tags, _tags(target_obj)):
            return -1.0
        score += 1.0
    elif rule.target_tags:
        tgt_tags = _tags(target_ent) if target_ent else _tags(target_obj) if target_obj else set()
        if not _has_any(rule.target_tags, tgt_tags):
            return -1.0
        score += 0.5

    return score


def _substitute_effects(
    effects: list[TransitionProposal],
    *,
    actor_id: str,
    target_id: Optional[str],
    tool_id: Optional[str],
    actor: EntityState,
) -> list[TransitionProposal]:
    out: list[TransitionProposal] = []
    for eff in effects:
        payload = dict(eff.payload or {})
        for key, val in list(payload.items()):
            if val == "$actor":
                payload[key] = actor_id
            elif val == "$target" and target_id:
                payload[key] = target_id
            elif val == "$tool" and tool_id:
                payload[key] = tool_id
            elif key in ("x", "y") and val == "$actor_x":
                payload[key] = actor.position.x
            elif key in ("x", "y") and val == "$actor_y":
                payload[key] = actor.position.y
        out.append(TransitionProposal(kind=eff.kind, payload=payload))
    return out


def _resolve_by_tag(world, actor, hint: str):
    """Match role words like 'bartender' to entity tags."""
    hint_l = hint.lower()
    visible = visible_from(world.spatial, actor.position, actor.sight_range)
    for eid, ent in world.spatial.entities.items():
        if not ent.alive or ent.entity_id == actor.entity_id:
            continue
        if ent.position not in visible:
            continue
        tags = {t.lower() for t in (ent.tags or [])}
        if hint_l in tags or hint_l in ent.name.lower():
            return eid, ent
    return None


def _entity_named_in_text(world: WorldState, actor: EntityState, text: str) -> Optional[tuple[EntityId, EntityState]]:
    lower = text.lower()
    visible = visible_from(world.spatial, actor.position, actor.sight_range)
    # Role/tag hints: "the bartender", "a guard"
    for word in re.split(r"[\s\-_']+", lower):
        if word in ("the", "a", "an", "at", "to", "with"):
            continue
        tagged = _resolve_by_tag(world, actor, word)
        if tagged:
            return tagged
    best: Optional[tuple[EntityId, EntityState, int]] = None
    for eid, ent in world.spatial.entities.items():
        if not ent.alive or ent.entity_id == actor.entity_id:
            continue
        if ent.position not in visible:
            continue
        name_l = ent.name.lower()
        # Match full name or distinctive token (skip "the", "a")
        tokens = [t for t in name_l.split() if t not in ("the", "a", "an")]
        if name_l in lower or any(t in lower for t in tokens if len(t) > 3):
            dist = actor.position.manhattan(ent.position)
            if best is None or dist < best[2]:
                best = (eid, ent, dist)
    return (best[0], best[1]) if best else None


def match_interaction_rule(
    world: WorldState,
    actor_id: EntityId,
    intent: str,
) -> Optional[tuple[InteractionRule, EntityState, Optional[EntityState], Optional[ObjectState], Optional[ObjectState]]]:
    actor = world.spatial.entities.get(actor_id)
    if actor is None or not world.config.interaction_rules:
        return None

    slots = extract_interaction_slots(intent)
    target_ent_pair = _resolve_visible_entity(world, actor, slots.target_name)
    if target_ent_pair is None:
        target_ent_pair = _entity_named_in_text(world, actor, intent)
    target_ent = target_ent_pair[1] if target_ent_pair else None
    target_obj = _resolve_visible_object(world, actor, slots.target_name)
    tool = _resolve_tool(actor, world, [], slots.tool_name)

    best: Optional[tuple[float, InteractionRule]] = None
    for rule in world.config.interaction_rules:
        score = _score_rule(rule, slots, actor, target_ent, target_obj, tool)
        if score < 0:
            continue
        if tool is None and rule.tool_tags:
            tool = _resolve_tool(actor, world, rule.tool_tags, slots.tool_name)
            score = _score_rule(rule, slots, actor, target_ent, target_obj, tool)
            if score < 0:
                continue
        if best is None or score > best[0]:
            best = (score, rule)

    if best is None:
        return None
    rule = best[1]
    if rule.tool_tags and tool is None:
        tool = _resolve_tool(actor, world, rule.tool_tags, slots.tool_name)
    return rule, actor, target_ent, target_obj, tool


def resolve_interaction_action(
    world: WorldState,
    actor_id: EntityId,
    intent: str,
) -> Optional[SemanticAction]:
    """
    Try to resolve player intent via interaction grammar.

    Returns a SemanticAction with proposed_effects when a rule matches.
    """
    matched = match_interaction_rule(world, actor_id, intent)
    if matched is None:
        return None

    rule, actor, target_ent, target_obj, tool = matched
    target: Optional[str] = None
    if rule.target_kind.lower() == "entity" and target_ent is not None:
        target = str(target_ent.entity_id)
    elif target_obj is not None:
        target = str(target_obj.object_id)
    elif target_ent is not None:
        target = str(target_ent.entity_id)

    target_id = str(target) if target else None
    tool_id = str(tool.object_id) if tool else None
    effects = _substitute_effects(
        list(rule.effects),
        actor_id=str(actor_id),
        target_id=target_id,
        tool_id=tool_id,
        actor=actor,
    )

    contest = rule.contest

    return SemanticAction(
        verb=rule.verb,
        actor=actor_id,
        target=target,
        intent=IntentBlock(
            rationale="",
            manner=rule.name or rule.id,
            desired_outcome=[rule.verb],
        ),
        style=StyleBlock(aggression=30, visibility=70),
        proposed_effects=effects,
        contest=contest,
        raw_input=intent,
    )


def interaction_hints_for_actor(
    world: WorldState,
    actor_id: EntityId,
    *,
    max_hints: int = 10,
) -> list[Affordance]:
    """Build projection/menu hints from matchable interaction rules."""
    actor = world.spatial.entities.get(actor_id)
    if actor is None:
        return []

    hints: list[Affordance] = []
    seen: set[str] = set()
    visible = visible_from(world.spatial, actor.position, actor.sight_range)
    inv_tags: set[str] = set()
    for obj in _inventory_objects(actor, world):
        inv_tags.update(_tags(obj))

    for rule in sorted(world.config.interaction_rules, key=lambda r: -r.priority):
        if not _has_any(rule.actor_tags, _tags(actor)) and rule.actor_tags:
            continue
        if rule.tool_tags and not _has_any(rule.tool_tags, inv_tags):
            continue

        label = rule.name or rule.id.replace("_", " ")
        if label in seen:
            continue

        if rule.target_kind.lower() == "entity":
            for ent in world.spatial.entities.values():
                if not ent.alive or ent.entity_id == actor_id:
                    continue
                if ent.position not in visible:
                    continue
                if rule.target_tags and not _has_any(rule.target_tags, _tags(ent)):
                    continue
                desc = f"{label} → {ent.name}"
                if desc not in seen:
                    seen.add(desc)
                    hints.append(Affordance(verb=rule.verb, description=desc, targets=[str(ent.entity_id)]))
                break
        elif rule.target_kind.lower() == "object":
            for obj in world.spatial.objects.values():
                if obj.position and obj.position not in visible:
                    continue
                if rule.target_tags and not _has_any(rule.target_tags, _tags(obj)):
                    continue
                desc = f"{label} → {obj.name}"
                if desc not in seen:
                    seen.add(desc)
                    hints.append(Affordance(verb=rule.verb, description=desc, targets=[str(obj.object_id)]))
                break
        else:
            seen.add(label)
            hints.append(Affordance(verb=rule.verb, description=label, targets=[]))

        if len(hints) >= max_hints:
            break

    return hints


def interaction_hint_strings(world: WorldState, actor_id: EntityId, *, max_hints: int = 8) -> list[str]:
    return [a.description for a in interaction_hints_for_actor(world, actor_id, max_hints=max_hints)]
