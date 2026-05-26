"""
Deterministic adjudication templates for open-ended Zone 3 actions.

Matched before the generic rule-based fallback in ``world_adjudicator``.
Covers trade, reputation, faction standing, and improvised item creation.
"""

from __future__ import annotations

import re
import uuid
from typing import Optional

from .schemas import (
    AdjudicationResult,
    AlertnessLevel,
    EdgeKind,
    EmotionalState,
    EntityId,
    GroundingResult,
    ScheduledEffect,
    ScheduledEffectKind,
    SemanticAction,
    TransitionKind,
    TransitionProposal,
    ValidationResult,
    WorldFact,
    WorldFactScope,
    WorldState,
    coord_key,
)

# ── Intent patterns ──────────────────────────────────────────────────────────

_BUY_RE = re.compile(
    r"(?:buy|purchase|order)\s+(?:a\s+|an\s+|the\s+)?(.+?)"
    r"(?:\s+for|\s+with|\s+at)\s+(?:(\d+)\s*gold|(\d+)\s*coins?)?",
    re.I,
)
_SELL_RE = re.compile(
    r"(?:sell|offer)\s+(?:my\s+|the\s+|a\s+)?(.+?)"
    r"(?:\s+for|\s+to|\s+at)\s+(?:(\d+)\s*gold)?",
    re.I,
)
_HAGGLE_RE = re.compile(r"\b(haggle|barter|negotiate|bargain)\b", re.I)
_REPUTATION_POS_RE = re.compile(
    r"\b(praise|commend|vouch for|speak well of|recommend)\b", re.I,
)
_REPUTATION_NEG_RE = re.compile(
    r"\b(slander|defame|smear|accuse|spread rumors? about|badmouth)\b", re.I,
)
_FACTION_JOIN_RE = re.compile(
    r"\b(join|pledge(?:\s+allegiance)?|swear(?:\s+fealty)?|align with)\b"
    r".{0,40}\b(faction|guild|order|brotherhood|company|crew)\b",
    re.I,
)
_CRAFT_RE = re.compile(
    r"\b(craft|make|forge|brew|assemble|fashion|whittle|knit)\b"
    r"\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:\s+from|\s+using|\s+with|\s*$)",
    re.I,
)
_REST_RE = re.compile(r"\b(rest|sleep|camp|nap|doze|lie down|recover strength)\b", re.I)
_INTIMIDATE_RE = re.compile(
    r"\b(intimidate|threaten|menace|frighten|bully|cow|warn)\b", re.I,
)
_BRIBE_RE = re.compile(
    r"\b(bribe|pay off|grease palms|slip coins|buy silence)\b", re.I,
)
_DISGUISE_RE = re.compile(
    r"\b(disguise|pretend to be|impersonate|dress as|pose as|pass as)\b", re.I,
)
_SEARCH_RE = re.compile(
    r"\b(search|investigate|examine closely|inspect|rummage|scour|look for)\b",
    re.I,
)
_DESTROY_RE = re.compile(
    r"\b(destroy|smash|demolish|break apart|obliterate|burn down|tear down)\b",
    re.I,
)
_MESSAGE_RE = re.compile(
    r"\b(?:leave|write|pin|post)\s+(?:a\s+)?(?:note|message|notice)"
    r"(?:\s+(?:that\s+)?(?:says?|reading)?[:\s]+)?(.+)?$",
    re.I,
)
_HIRE_RE = re.compile(
    r"\b(hire|employ|recruit|commission|pay someone to)\b", re.I,
)
_TRAVEL_RE = re.compile(
    r"\b(travel to|journey to|head to|make for|set out for|ride to|sail to)\b"
    r"\s+(?:the\s+)?(.+?)(?:\.|$)",
    re.I,
)
_GOLD_AMOUNT_RE = re.compile(r"(\d+)\s*(?:gold|coins?)", re.I)


def _target_entity_id(action: SemanticAction, world: WorldState) -> Optional[str]:
    if isinstance(action.target, str):
        tid = str(action.target)
        if EntityId(tid) in world.spatial.entities:
            return tid
    return None


def _primary_tag_for_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower().strip())[:32] or "goods"


def _parse_gold_amount(text: str, default: float = 10.0) -> float:
    m = _GOLD_AMOUNT_RE.search(text)
    if m:
        return max(1.0, float(m.group(1)))
    return default


def _schedule_ripple(
    adj: AdjudicationResult,
    world: WorldState,
    actor_id: str,
    narration: str,
    *,
    offset: int = 2,
    rationale: str = "adjudicated action ripple",
) -> None:
    tick = world.tick
    adj.scheduled_effects.append(
        ScheduledEffect(
            fire_tick=tick + offset,
            created_tick=tick,
            source_actor=actor_id,
            kind=ScheduledEffectKind.NARRATION,
            narration=narration[:240],
            rationale=rationale,
        )
    )


def _tile_mark_proposal(actor, mark: str) -> TransitionProposal:
    mark = re.sub(r"\s+", " ", mark.strip())[:60] or "altered"
    return TransitionProposal(
        kind=TransitionKind.TILE_MARKED.value,
        payload={
            "x": actor.position.x,
            "y": actor.position.y,
            "mark": mark,
        },
    )


def try_adjudication_templates(
    world: WorldState,
    action: SemanticAction,
    intent: str,
    grounding: GroundingResult,
    result: ValidationResult,
) -> Optional[AdjudicationResult]:
    """Return a full adjudication result when a template matches, else None."""
    text = (intent or action.raw_input or "").strip()
    if not text:
        return None

    for fn in (
        _template_trade,
        _template_reputation,
        _template_faction,
        _template_craft,
        _template_rest,
        _template_intimidate,
        _template_bribe,
        _template_disguise,
        _template_search,
        _template_destroy,
        _template_message,
        _template_hire,
        _template_travel,
    ):
        adj = fn(world, action, text, grounding, result)
        if adj is not None:
            return adj
    return None


def _template_trade(
    world: WorldState,
    action: SemanticAction,
    text: str,
    _grounding: GroundingResult,
    _result: ValidationResult,
) -> Optional[AdjudicationResult]:
    from .economy import get_market_price, record_trade

    lower = text.lower()
    actor_id = str(action.actor)
    actor = world.spatial.entities.get(action.actor)
    if actor is None:
        return None

    target_id = _target_entity_id(action, world)
    m_buy = _BUY_RE.search(text)
    m_sell = _SELL_RE.search(text)
    m_haggle = _HAGGLE_RE.search(text)

    if not (m_buy or m_sell or m_haggle):
        return None

    item_name = ""
    gold_amount = 0.0
    if m_buy:
        item_name = m_buy.group(1).strip()
        if m_buy.group(2):
            gold_amount = float(m_buy.group(2))
        elif m_buy.group(3):
            gold_amount = float(m_buy.group(3))
    elif m_sell:
        item_name = m_sell.group(1).strip()
        if m_sell.group(2):
            gold_amount = float(m_sell.group(2))
    else:
        item_name = "goods"

    tag = _primary_tag_for_name(item_name)
    if gold_amount <= 0:
        gold_amount = get_market_price(world, tag, region_id=world.active_region_id)

    if m_haggle and gold_amount > 0:
        gold_amount = max(1.0, round(gold_amount * 0.85, 1))

    actor_gold = float(actor.stats.get("gold", 0))
    if gold_amount > actor_gold:
        return AdjudicationResult(
            ruling_text=(
                f"You cannot afford {item_name} — it would cost {gold_amount:.0f} gold "
                f"and you only have {actor_gold:.0f}."
            ),
        )

    adj = AdjudicationResult(
        ruling_text=(
            f"The deal is struck: {item_name} for {gold_amount:.0f} gold."
            if not m_haggle
            else f"After haggling, you settle on {item_name} for {gold_amount:.0f} gold."
        ),
    )

    adj.transition_proposals.extend([
        TransitionProposal(
            kind=TransitionKind.ENTITY_STAT_CHANGED.value,
            payload={
                "entity_id": actor_id,
                "stat": "gold",
                "delta": -gold_amount,
                "cause": "adjudicated_trade",
                "actor": actor_id,
            },
        ),
        TransitionProposal(
            kind=TransitionKind.ENTITY_STAT_CHANGED.value,
            payload={
                "entity_id": actor_id,
                "stat": "reputation",
                "delta": 1.5,
                "cause": "completed_trade",
                "actor": actor_id,
            },
        ),
    ])

    if target_id:
        adj.transition_proposals.append(
            TransitionProposal(
                kind=TransitionKind.ENTITY_STAT_CHANGED.value,
                payload={
                    "entity_id": target_id,
                    "stat": "gold",
                    "delta": gold_amount,
                    "cause": "adjudicated_trade",
                    "actor": actor_id,
                },
            )
        )
        adj.transition_proposals.append(
            TransitionProposal(
                kind=TransitionKind.EDGE_UPDATED.value,
                payload={
                    "source": actor_id,
                    "target": target_id,
                    "edge_kind": EdgeKind.INTERACTED.value,
                    "delta": 0.12,
                    "meta": {"verb": "trade", "cause": "adjudication"},
                },
            )
        )
        target_name = world.spatial.entities[EntityId(target_id)].name
        adj.facts.append(
            WorldFact(
                claim=f"{actor.name} traded with {target_name} for {item_name}.",
                scope=WorldFactScope.ENTITY,
                subject_id=target_id,
                established_tick=world.tick,
                established_by=actor_id,
                tags=["trade", "adjudicated"],
            )
        )
    else:
        adj.facts.append(
            WorldFact(
                claim=f"{actor.name} acquired {item_name} for {gold_amount:.0f} gold.",
                scope=WorldFactScope.TILE,
                subject_id=coord_key(actor.position),
                established_tick=world.tick,
                established_by=actor_id,
                tags=["trade", "adjudicated"],
            )
        )

    record_trade(world, tag, quantity=1, region_id=world.active_region_id)
    return adj


def _template_reputation(
    world: WorldState,
    action: SemanticAction,
    text: str,
    _grounding: GroundingResult,
    _result: ValidationResult,
) -> Optional[AdjudicationResult]:
    actor_id = str(action.actor)
    actor = world.spatial.entities.get(action.actor)
    if actor is None:
        return None

    pos = _REPUTATION_POS_RE.search(text)
    neg = _REPUTATION_NEG_RE.search(text)
    if not pos and not neg:
        return None

    target_id = _target_entity_id(action, world)
    delta = 5.0 if pos else -8.0
    verb_label = "praised" if pos else "slandered"

    adj = AdjudicationResult()
    if target_id:
        target = world.spatial.entities[EntityId(target_id)]
        adj.ruling_text = (
            f"Word spreads that you {verb_label} {target.name} — "
            f"reputation shifts accordingly."
        )
        adj.transition_proposals.extend([
            TransitionProposal(
                kind=TransitionKind.ENTITY_STAT_CHANGED.value,
                payload={
                    "entity_id": target_id,
                    "stat": "reputation",
                    "delta": delta,
                    "cause": "adjudicated_reputation",
                    "actor": actor_id,
                },
            ),
            TransitionProposal(
                kind=TransitionKind.EDGE_UPDATED.value,
                payload={
                    "source": actor_id,
                    "target": target_id,
                    "edge_kind": EdgeKind.WITNESSED.value,
                    "delta": 0.1 if pos else -0.05,
                    "meta": {"cause": verb_label, "verb": "reputation"},
                },
            ),
        ])
        adj.facts.append(
            WorldFact(
                claim=f"{actor.name} {verb_label} {target.name}.",
                scope=WorldFactScope.ENTITY,
                subject_id=target_id,
                established_tick=world.tick,
                established_by=actor_id,
                tags=["reputation", "adjudicated"],
            )
        )
    else:
        adj.ruling_text = (
            f"Your words change how others see you — reputation "
            f"{'rises' if pos else 'falls'}."
        )
        adj.transition_proposals.append(
            TransitionProposal(
                kind=TransitionKind.ENTITY_STAT_CHANGED.value,
                payload={
                    "entity_id": actor_id,
                    "stat": "reputation",
                    "delta": delta * 0.5,
                    "cause": "adjudicated_reputation",
                    "actor": actor_id,
                },
            )
        )
        adj.facts.append(
            WorldFact(
                claim=f"{actor.name} {verb_label} someone in public.",
                scope=WorldFactScope.TILE,
                subject_id=coord_key(actor.position),
                established_tick=world.tick,
                established_by=actor_id,
                tags=["reputation", "adjudicated"],
            )
        )

    adj.scheduled_effects.append(
        ScheduledEffect(
            fire_tick=world.tick + 2,
            created_tick=world.tick,
            source_actor=actor_id,
            kind=ScheduledEffectKind.NARRATION,
            narration=(
                f"Rumours about what {actor.name} said begin to circulate."
            ),
            rationale="reputation ripple from adjudicated speech",
        )
    )
    return adj


def _template_faction(
    world: WorldState,
    action: SemanticAction,
    text: str,
    _grounding: GroundingResult,
    _result: ValidationResult,
) -> Optional[AdjudicationResult]:
    if not _FACTION_JOIN_RE.search(text):
        return None

    actor_id = str(action.actor)
    actor = world.spatial.entities.get(action.actor)
    if actor is None:
        return None

    faction_id = actor.faction or ""
    if not faction_id:
        factions = world.meta.get("factions") or {}
        if len(factions) == 1:
            faction_id = next(iter(factions.keys()))
        elif isinstance(action.target, str) and action.target in factions:
            faction_id = action.target
        else:
            for fid, fdata in factions.items():
                fname = str(fdata.get("name", "")).lower()
                if fname and fname in text.lower():
                    faction_id = fid
                    break

    if not faction_id:
        return AdjudicationResult(
            ruling_text=(
                "You declare your allegiance, but no faction here "
                "recognizes the pledge yet."
            ),
            facts=[
                WorldFact(
                    claim=f"{actor.name} attempted to pledge to a faction.",
                    scope=WorldFactScope.WORLD,
                    established_tick=world.tick,
                    established_by=actor_id,
                    tags=["faction", "adjudicated"],
                ),
            ],
        )

    fname = (world.meta.get("factions") or {}).get(faction_id, {}).get(
        "name", faction_id
    )
    adj = AdjudicationResult(
        ruling_text=f"You are recognized as aligned with {fname}.",
    )
    adj.transition_proposals.append(
        TransitionProposal(
            kind=TransitionKind.EDGE_UPDATED.value,
            payload={
                "source": actor_id,
                "target": faction_id,
                "edge_kind": EdgeKind.FACTION_MEMBER.value,
                "delta": 0.75,
                "meta": {"cause": "adjudicated_oath", "verb": "join_faction"},
            },
        )
    )
    adj.facts.append(
        WorldFact(
            claim=f"{actor.name} pledged allegiance to {fname}.",
            scope=WorldFactScope.ENTITY,
            subject_id=actor_id,
            established_tick=world.tick,
            established_by=actor_id,
            tags=["faction", "adjudicated"],
        )
    )
    return adj


def _template_craft(
    world: WorldState,
    action: SemanticAction,
    text: str,
    _grounding: GroundingResult,
    _result: ValidationResult,
) -> Optional[AdjudicationResult]:
    m = _CRAFT_RE.search(text)
    if not m:
        return None

    actor_id = str(action.actor)
    actor = world.spatial.entities.get(action.actor)
    if actor is None:
        return None

    item_name = m.group(1).strip()[:40] or "handmade item"
    new_oid = f"obj_{uuid.uuid4().hex[:8]}"
    tag = _primary_tag_for_name(item_name)

    adj = AdjudicationResult(
        ruling_text=f"You fashion {item_name} — it joins your belongings.",
    )
    adj.transition_proposals.extend([
        TransitionProposal(
            kind=TransitionKind.ENTITY_STAT_CHANGED.value,
            payload={
                "entity_id": actor_id,
                "stat": "stamina",
                "delta": -8.0,
                "cause": "adjudicated_craft",
                "actor": actor_id,
            },
        ),
        TransitionProposal(
            kind=TransitionKind.ITEM_SYNTHESIZED.value,
            payload={
                "object_id": new_oid,
                "name": item_name,
                "tags": [tag, "crafted"],
                "weight": 1.0,
                "bulk": 1.0,
                "attributes": {"crafted": True, "value": 5},
                "owner_entity_id": actor_id,
                "cause": "adjudicated_craft",
            },
        ),
    ])
    adj.facts.append(
        WorldFact(
            claim=f"{actor.name} crafted {item_name}.",
            scope=WorldFactScope.TILE,
            subject_id=coord_key(actor.position),
            established_tick=world.tick,
            established_by=actor_id,
            tags=["craft", "adjudicated"],
        )
    )
    return adj


def _template_rest(
    world: WorldState,
    action: SemanticAction,
    text: str,
    _grounding: GroundingResult,
    _result: ValidationResult,
) -> Optional[AdjudicationResult]:
    if not _REST_RE.search(text):
        return None

    actor_id = str(action.actor)
    actor = world.spatial.entities.get(action.actor)
    if actor is None:
        return None

    is_sleep = bool(re.search(r"\b(sleep|doze|nap)\b", text, re.I))
    fatigue_gain = 70.0 if is_sleep else 40.0
    stamina_gain = 15.0 if is_sleep else 8.0

    adj = AdjudicationResult(
        ruling_text=(
            "You sleep deeply and wake somewhat restored."
            if is_sleep
            else "You catch your breath and recover some strength."
        ),
    )
    adj.transition_proposals.extend([
        TransitionProposal(
            kind=TransitionKind.NEED_CHANGED.value,
            payload={
                "entity_id": actor_id,
                "need": "fatigue",
                "delta": fatigue_gain,
                "cause": "adjudicated_rest",
                "actor": actor_id,
            },
        ),
        TransitionProposal(
            kind=TransitionKind.ENTITY_STAT_CHANGED.value,
            payload={
                "entity_id": actor_id,
                "stat": "stamina",
                "delta": stamina_gain,
                "cause": "adjudicated_rest",
                "actor": actor_id,
            },
        ),
    ])
    adj.facts.append(
        WorldFact(
            claim=f"{actor.name} rested here.",
            scope=WorldFactScope.TILE,
            subject_id=coord_key(actor.position),
            established_tick=world.tick,
            established_by=actor_id,
            tags=["rest", "adjudicated"],
        )
    )
    return adj


def _template_intimidate(
    world: WorldState,
    action: SemanticAction,
    text: str,
    _grounding: GroundingResult,
    _result: ValidationResult,
) -> Optional[AdjudicationResult]:
    if not _INTIMIDATE_RE.search(text):
        return None

    actor_id = str(action.actor)
    actor = world.spatial.entities.get(action.actor)
    if actor is None:
        return None

    target_id = _target_entity_id(action, world)
    adj = AdjudicationResult()

    if target_id:
        target = world.spatial.entities[EntityId(target_id)]
        adj.ruling_text = (
            f"You loom over {target.name} — fear and anger flash across their face."
        )
        adj.transition_proposals.extend([
            TransitionProposal(
                kind=TransitionKind.ENTITY_ALERTNESS_CHANGED.value,
                payload={
                    "entity_id": target_id,
                    "to": AlertnessLevel.HIGH.value,
                    "cause": "adjudicated_intimidate",
                    "actor": actor_id,
                },
            ),
            TransitionProposal(
                kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED.value,
                payload={
                    "entity_id": target_id,
                    "to": EmotionalState.FEARFUL.value,
                    "cause": "adjudicated_intimidate",
                    "actor": actor_id,
                },
            ),
            TransitionProposal(
                kind=TransitionKind.EDGE_UPDATED.value,
                payload={
                    "source": target_id,
                    "target": actor_id,
                    "edge_kind": EdgeKind.FEARS.value,
                    "delta": 0.2,
                    "meta": {"verb": "intimidate", "cause": "adjudication"},
                },
            ),
            TransitionProposal(
                kind=TransitionKind.EDGE_UPDATED.value,
                payload={
                    "source": actor_id,
                    "target": target_id,
                    "edge_kind": EdgeKind.THREATENED.value,
                    "delta": 0.25,
                    "meta": {"verb": "intimidate", "cause": "adjudication"},
                },
            ),
        ])
        adj.facts.append(
            WorldFact(
                claim=f"{actor.name} intimidated {target.name}.",
                scope=WorldFactScope.ENTITY,
                subject_id=target_id,
                established_tick=world.tick,
                established_by=actor_id,
                tags=["intimidation", "violence", "adjudicated"],
            )
        )
    else:
        adj.ruling_text = (
            "Your threats hang in the air — those nearby take notice."
        )
        adj.transition_proposals.append(
            TransitionProposal(
                kind=TransitionKind.NOISE_EVENT.value,
                payload={
                    "source_id": actor_id,
                    "noise_level": 45,
                    "cause": "adjudicated_intimidate",
                },
            )
        )
        adj.facts.append(
            WorldFact(
                claim=f"{actor.name} made threatening gestures in public.",
                scope=WorldFactScope.TILE,
                subject_id=coord_key(actor.position),
                established_tick=world.tick,
                established_by=actor_id,
                tags=["intimidation", "adjudicated"],
            )
        )

    _schedule_ripple(
        adj, world, actor_id,
        adj.ruling_text,
        rationale="intimidation witnessed",
    )
    return adj


def _template_bribe(
    world: WorldState,
    action: SemanticAction,
    text: str,
    _grounding: GroundingResult,
    _result: ValidationResult,
) -> Optional[AdjudicationResult]:
    if not _BRIBE_RE.search(text):
        return None

    actor_id = str(action.actor)
    actor = world.spatial.entities.get(action.actor)
    if actor is None:
        return None

    target_id = _target_entity_id(action, world)
    gold_amount = _parse_gold_amount(text, default=15.0)
    actor_gold = float(actor.stats.get("gold", 0))
    if gold_amount > actor_gold:
        return AdjudicationResult(
            ruling_text=(
                f"You don't have enough gold for a bribe "
                f"({gold_amount:.0f} needed, {actor_gold:.0f} on hand)."
            ),
        )

    adj = AdjudicationResult(
        ruling_text=f"You slip {gold_amount:.0f} gold across — eyes meet, understanding passes.",
    )
    adj.transition_proposals.append(
        TransitionProposal(
            kind=TransitionKind.ENTITY_STAT_CHANGED.value,
            payload={
                "entity_id": actor_id,
                "stat": "gold",
                "delta": -gold_amount,
                "cause": "adjudicated_bribe",
                "actor": actor_id,
            },
        )
    )
    if target_id:
        target = world.spatial.entities[EntityId(target_id)]
        adj.transition_proposals.extend([
            TransitionProposal(
                kind=TransitionKind.ENTITY_STAT_CHANGED.value,
                payload={
                    "entity_id": target_id,
                    "stat": "gold",
                    "delta": gold_amount,
                    "cause": "adjudicated_bribe",
                    "actor": actor_id,
                },
            ),
            TransitionProposal(
                kind=TransitionKind.EDGE_UPDATED.value,
                payload={
                    "source": actor_id,
                    "target": target_id,
                    "edge_kind": EdgeKind.OWES_DEBT.value,
                    "delta": 0.15,
                    "meta": {"verb": "bribe", "cause": "adjudication", "gold": gold_amount},
                },
            ),
        ])
        adj.facts.append(
            WorldFact(
                claim=f"{actor.name} bribed {target.name} with {gold_amount:.0f} gold.",
                scope=WorldFactScope.ENTITY,
                subject_id=target_id,
                established_tick=world.tick,
                established_by=actor_id,
                tags=["bribe", "adjudicated"],
            )
        )
    else:
        adj.facts.append(
            WorldFact(
                claim=f"{actor.name} paid a discreet bribe ({gold_amount:.0f} gold).",
                scope=WorldFactScope.TILE,
                subject_id=coord_key(actor.position),
                established_tick=world.tick,
                established_by=actor_id,
                tags=["bribe", "adjudicated"],
            )
        )

    _schedule_ripple(
        adj, world, actor_id,
        "Someone may have seen the exchange of coin.",
        rationale="bribe ripple",
    )
    return adj


def _template_disguise(
    world: WorldState,
    action: SemanticAction,
    text: str,
    _grounding: GroundingResult,
    _result: ValidationResult,
) -> Optional[AdjudicationResult]:
    if not _DISGUISE_RE.search(text):
        return None

    actor_id = str(action.actor)
    actor = world.spatial.entities.get(action.actor)
    if actor is None:
        return None

    identity = "someone else"
    m = re.search(
        r"(?:as|like)\s+(?:a\s+|an\s+|the\s+)?([a-z][\w\s]{2,30})",
        text,
        re.I,
    )
    if m:
        identity = m.group(1).strip()

    adj = AdjudicationResult(
        ruling_text=f"You adopt a new appearance — passing as {identity}.",
    )
    adj.transition_proposals.append(
        TransitionProposal(
            kind=TransitionKind.ENTITY_PROPERTY_CHANGED.value,
            payload={
                "entity_id": actor_id,
                "prop": "disguise",
                "new_value": identity[:60],
                "cause": "adjudicated_disguise",
                "actor": actor_id,
            },
        )
    )
    adj.facts.append(
        WorldFact(
            claim=f"{actor.name} is disguised as {identity}.",
            scope=WorldFactScope.ENTITY,
            subject_id=actor_id,
            established_tick=world.tick,
            established_by=actor_id,
            tags=["disguise", "adjudicated"],
        )
    )
    return adj


def _template_search(
    world: WorldState,
    action: SemanticAction,
    text: str,
    _grounding: GroundingResult,
    _result: ValidationResult,
) -> Optional[AdjudicationResult]:
    if not _SEARCH_RE.search(text):
        return None

    actor_id = str(action.actor)
    actor = world.spatial.entities.get(action.actor)
    if actor is None:
        return None

    target_id = _target_entity_id(action, world)
    subject = ""
    if target_id:
        target = world.spatial.entities[EntityId(target_id)]
        subject = target.name
        if target.secrets:
            hint = target.secrets[0][:80]
            adj = AdjudicationResult(
                ruling_text=f"Searching {subject}, you find something suspicious: a hidden detail.",
            )
            adj.facts.append(
                WorldFact(
                    claim=f"{actor.name} discovered a secret about {subject}.",
                    scope=WorldFactScope.ENTITY,
                    subject_id=target_id,
                    established_tick=world.tick,
                    established_by=actor_id,
                    tags=["search", "investigation", "adjudicated"],
                    meta={"hint": hint},
                )
            )
        else:
            adj = AdjudicationResult(
                ruling_text=f"You search {subject} thoroughly but find nothing unusual.",
            )
            adj.facts.append(
                WorldFact(
                    claim=f"{actor.name} searched {subject} and found nothing.",
                    scope=WorldFactScope.ENTITY,
                    subject_id=target_id,
                    established_tick=world.tick,
                    established_by=actor_id,
                    tags=["search", "adjudicated"],
                )
            )
    else:
        area = "the area"
        m = re.search(r"(?:for|in|around)\s+(?:the\s+)?(.+?)(?:\.|$)", text, re.I)
        if m:
            area = m.group(1).strip()[:40]
        adj = AdjudicationResult(
            ruling_text=f"You search {area} — traces remain where you looked.",
        )
        mark = f"searched: {area}"[:60]
        adj.transition_proposals.append(_tile_mark_proposal(actor, mark))
        adj.facts.append(
            WorldFact(
                claim=f"{actor.name} searched {area}.",
                scope=WorldFactScope.TILE,
                subject_id=coord_key(actor.position),
                established_tick=world.tick,
                established_by=actor_id,
                tags=["search", "investigation", "adjudicated"],
            )
        )

    adj.transition_proposals.append(
        TransitionProposal(
            kind=TransitionKind.ENTITY_STAT_CHANGED.value,
            payload={
                "entity_id": actor_id,
                "stat": "stamina",
                "delta": -5.0,
                "cause": "adjudicated_search",
                "actor": actor_id,
            },
        )
    )
    return adj


def _template_destroy(
    world: WorldState,
    action: SemanticAction,
    text: str,
    _grounding: GroundingResult,
    _result: ValidationResult,
) -> Optional[AdjudicationResult]:
    if not _DESTROY_RE.search(text):
        return None

    actor_id = str(action.actor)
    actor = world.spatial.entities.get(action.actor)
    if actor is None:
        return None

    target = "something nearby"
    m = re.search(
        r"(?:destroy|smash|demolish|break|obliterate|burn down|tear down)\s+"
        r"(?:the\s+|a\s+|an\s+)?(.+?)(?:\.|$)",
        text,
        re.I,
    )
    if m:
        target = m.group(1).strip()[:40]

    mark = f"destroyed: {target}"[:60]
    adj = AdjudicationResult(
        ruling_text=f"You destroy {target} — debris and damage remain.",
    )
    adj.transition_proposals.extend([
        _tile_mark_proposal(actor, mark),
        TransitionProposal(
            kind=TransitionKind.ENVIRONMENT_STATE_CHANGED.value,
            payload={
                "x": actor.position.x,
                "y": actor.position.y,
                "key": "damaged",
                "value": True,
                "cause": "adjudicated_destroy",
                "author": actor_id,
            },
        ),
        TransitionProposal(
            kind=TransitionKind.NOISE_EVENT.value,
            payload={
                "source_id": actor_id,
                "noise_level": 65,
                "cause": "adjudicated_destroy",
            },
        ),
        TransitionProposal(
            kind=TransitionKind.ENTITY_STAT_CHANGED.value,
            payload={
                "entity_id": actor_id,
                "stat": "stamina",
                "delta": -12.0,
                "cause": "adjudicated_destroy",
                "actor": actor_id,
            },
        ),
    ])
    adj.facts.append(
        WorldFact(
            claim=f"{actor.name} destroyed {target}.",
            scope=WorldFactScope.TILE,
            subject_id=coord_key(actor.position),
            established_tick=world.tick,
            established_by=actor_id,
            tags=["environment", "violence", "adjudicated"],
        )
    )
    _schedule_ripple(
        adj, world, actor_id,
        f"People nearby react to the destruction of {target}.",
        rationale="destruction ripple",
    )
    return adj


def _template_message(
    world: WorldState,
    action: SemanticAction,
    text: str,
    _grounding: GroundingResult,
    _result: ValidationResult,
) -> Optional[AdjudicationResult]:
    if not _MESSAGE_RE.search(text):
        return None

    actor_id = str(action.actor)
    actor = world.spatial.entities.get(action.actor)
    if actor is None:
        return None

    m = _MESSAGE_RE.search(text)
    body = (m.group(1).strip() if m and m.group(1) else "")[:80]
    if not body:
        body = text[:80]

    mark = f"message: {body}"[:60]
    adj = AdjudicationResult(
        ruling_text=f"You leave a message: \"{body}\".",
    )
    adj.transition_proposals.append(_tile_mark_proposal(actor, mark))
    adj.facts.append(
        WorldFact(
            claim=f"A message was left here: {body}",
            scope=WorldFactScope.TILE,
            subject_id=coord_key(actor.position),
            established_tick=world.tick,
            established_by=actor_id,
            tags=["message", "adjudicated"],
        )
    )
    _schedule_ripple(
        adj, world, actor_id,
        "Someone notices the message you left.",
        rationale="message discovered",
    )
    return adj


def _template_hire(
    world: WorldState,
    action: SemanticAction,
    text: str,
    _grounding: GroundingResult,
    _result: ValidationResult,
) -> Optional[AdjudicationResult]:
    if not _HIRE_RE.search(text):
        return None

    actor_id = str(action.actor)
    actor = world.spatial.entities.get(action.actor)
    if actor is None:
        return None

    target_id = _target_entity_id(action, world)
    gold_amount = _parse_gold_amount(text, default=20.0)
    actor_gold = float(actor.stats.get("gold", 0))
    if gold_amount > actor_gold:
        return AdjudicationResult(
            ruling_text=(
                f"You cannot afford to hire help "
                f"({gold_amount:.0f} gold needed, {actor_gold:.0f} on hand)."
            ),
        )

    task = "a task"
    m = re.search(r"\bto\s+(.+?)(?:\.|$)", text, re.I)
    if m:
        task = m.group(1).strip()[:60]

    adj = AdjudicationResult(
        ruling_text=f"You hire help for {gold_amount:.0f} gold — terms agreed for: {task}.",
    )
    adj.transition_proposals.append(
        TransitionProposal(
            kind=TransitionKind.ENTITY_STAT_CHANGED.value,
            payload={
                "entity_id": actor_id,
                "stat": "gold",
                "delta": -gold_amount,
                "cause": "adjudicated_hire",
                "actor": actor_id,
            },
        )
    )
    if target_id:
        target = world.spatial.entities[EntityId(target_id)]
        adj.transition_proposals.extend([
            TransitionProposal(
                kind=TransitionKind.EDGE_UPDATED.value,
                payload={
                    "source": actor_id,
                    "target": target_id,
                    "edge_kind": EdgeKind.EMPLOYS.value,
                    "delta": 0.5,
                    "meta": {"verb": "hire", "task": task, "cause": "adjudication"},
                },
            ),
            TransitionProposal(
                kind=TransitionKind.ENTITY_GOAL_ADDED.value,
                payload={
                    "entity_id": target_id,
                    "goal": f"Fulfill the job for {actor.name}: {task}",
                    "cause": "adjudicated_hire",
                    "priority": 0.75,
                },
            ),
        ])
        adj.facts.append(
            WorldFact(
                claim=f"{actor.name} hired {target.name} to {task}.",
                scope=WorldFactScope.ENTITY,
                subject_id=target_id,
                established_tick=world.tick,
                established_by=actor_id,
                tags=["hire", "adjudicated"],
            )
        )
    else:
        adj.facts.append(
            WorldFact(
                claim=f"{actor.name} hired someone to {task} for {gold_amount:.0f} gold.",
                scope=WorldFactScope.TILE,
                subject_id=coord_key(actor.position),
                established_tick=world.tick,
                established_by=actor_id,
                tags=["hire", "adjudicated"],
            )
        )
    return adj


def _template_travel(
    world: WorldState,
    action: SemanticAction,
    text: str,
    _grounding: GroundingResult,
    _result: ValidationResult,
) -> Optional[AdjudicationResult]:
    m = _TRAVEL_RE.search(text)
    if not m:
        return None

    actor_id = str(action.actor)
    actor = world.spatial.entities.get(action.actor)
    if actor is None:
        return None

    destination = m.group(1).strip()[:60] or "distant lands"
    adj = AdjudicationResult(
        ruling_text=f"You set out toward {destination} — the journey begins.",
    )
    adj.transition_proposals.append(
        TransitionProposal(
            kind=TransitionKind.ENTITY_STAT_CHANGED.value,
            payload={
                "entity_id": actor_id,
                "stat": "stamina",
                "delta": -10.0,
                "cause": "adjudicated_travel",
                "actor": actor_id,
            },
        )
    )
    adj.facts.append(
        WorldFact(
            claim=f"{actor.name} is travelling toward {destination}.",
            scope=WorldFactScope.REGION,
            subject_id=world.active_region_id,
            established_tick=world.tick,
            established_by=actor_id,
            tags=["travel", "adjudicated"],
        )
    )
    _schedule_ripple(
        adj, world, actor_id,
        f"The road toward {destination} stretches ahead.",
        offset=3,
        rationale="travel underway",
    )
    return adj


__all__ = ["try_adjudication_templates"]
