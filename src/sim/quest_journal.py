"""
Quest journal — player-facing aggregation of objectives and world reactions.

Combines pack-declared goals, emergent fact-driven concerns, nearby NPC
pursuits, and pending scheduled ripples into one compact view for REPL,
TUI, and presentation clients.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .schemas import EntityId, WorldState

_EMERGENT_TAGS = frozenset({
    "theft", "violence", "investigation", "environment", "death", "intimidation",
})
_MAX_ENTRIES = 10
_TAG_PRIORITY = ("death", "violence", "theft", "investigation", "environment", "intimidation")


class QuestJournalEntry(BaseModel):
    """One line in the player's quest / concerns journal."""

    entry_id: str
    source: str  # pack | emergent | npc | scheduled
    title: str
    detail: str = ""
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    tick: int = 0
    status: str = "active"  # active | completed | upcoming


def build_quest_journal(
    world: "WorldState",
    focal_entity_id: Optional["EntityId"] = None,
    *,
    max_entries: int = _MAX_ENTRIES,
) -> list[QuestJournalEntry]:
    """Build a ranked, deduplicated journal for the focal entity (usually player)."""
    from .memory_retrieval import _fact_relevant_to_focal, fact_importance
    from .region_utils import entity_or_none, iter_entities
    from .schemas import EdgeKind, EntityId, EntityKind

    if focal_entity_id is None:
        focal_entity_id = next(
            (eid for eid, e in world.spatial.entities.items() if e.kind == EntityKind.PLAYER),
            EntityId(""),
        )

    focal = entity_or_none(world, focal_entity_id)
    entries: list[QuestJournalEntry] = []
    seen_titles: set[str] = set()

    def _add(entry: QuestJournalEntry) -> None:
        key = entry.title.lower().strip()
        if key in seen_titles:
            return
        seen_titles.add(key)
        entries.append(entry)

    for goal in world.config.goals:
        if goal.completed:
            _add(
                QuestJournalEntry(
                    entry_id=f"pack:{goal.id}",
                    source="pack",
                    title=goal.title,
                    detail=goal.description,
                    priority=0.95,
                    tick=goal.completed_at or world.tick,
                    status="completed" if goal.completed else "active",
                )
            )
        else:
            _add(
                QuestJournalEntry(
                    entry_id=f"pack:{goal.id}",
                    source="pack",
                    title=goal.title,
                    detail=goal.description,
                    priority=0.95,
                    tick=world.tick,
                    status="active",
                )
            )

    if focal is not None:
        pid = str(focal_entity_id)
        for holder_id, targets in world.relational.edges.items():
            if holder_id == pid:
                continue
            holder = entity_or_none(world, EntityId(holder_id))
            if holder is None or not holder.alive:
                continue
            if holder.region_id != focal.region_id:
                continue
            if holder.position.manhattan(focal.position) > focal.sight_range + 6:
                continue
            hname = holder.name
            for subject_id, edges in targets.items():
                for edge in edges:
                    if edge.kind != EdgeKind.BELIEVES_CLAIM:
                        continue
                    claim = (edge.meta or {}).get("claim", "")
                    if not claim:
                        continue
                    if subject_id != pid and focal.name.split()[0].lower() not in claim.lower():
                        continue
                    fidelity = edge.weight
                    tag = "rumour" if fidelity < 0.85 else "belief"
                    _add(
                        QuestJournalEntry(
                            entry_id=f"rumour:{holder_id}:{hash(claim) & 0xFFFF:04x}",
                            source="emergent",
                            title=f"{hname} heard something ({tag})",
                            detail=claim[:160],
                            priority=0.72 if tag == "rumour" else 0.78,
                            tick=world.tick,
                        )
                    )

        scored_facts: list[tuple[float, object]] = []
        for fact in world.world_facts:
            if not _fact_relevant_to_focal(fact, world, str(focal_entity_id)):
                continue
            if not _EMERGENT_TAGS.intersection(fact.tags):
                continue
            scored_facts.append((fact_importance(fact, world), fact))
        scored_facts.sort(key=lambda x: x[0], reverse=True)
        for _, fact in scored_facts[:5]:
            tag = next((t for t in _TAG_PRIORITY if t in fact.tags), fact.tags[0] if fact.tags else "event")
            _add(
                QuestJournalEntry(
                    entry_id=f"fact:{fact.fact_id}",
                    source="emergent",
                    title=f"World remembers ({tag})",
                    detail=fact.claim[:160],
                    priority=0.75,
                    tick=fact.established_tick,
                )
            )

        for eid, ent, _grid in iter_entities(world):
            if not ent.alive or ent.kind != EntityKind.NPC:
                continue
            if ent.region_id != focal.region_id:
                continue
            if ent.position.manhattan(focal.position) > focal.sight_range + 4:
                continue
            for goal in ent.goals[:2]:
                _add(
                    QuestJournalEntry(
                        entry_id=f"npc:{eid}:{hash(goal) & 0xFFFF:04x}",
                        source="npc",
                        title=f"{ent.name} is pursuing something",
                        detail=goal[:160],
                        priority=0.55,
                        tick=world.tick,
                    )
                )

    for effect in world.scheduled_effects:
        if effect.fire_tick <= world.tick:
            continue
        if not effect.narration and not effect.transitions:
            continue
        detail = (effect.narration or effect.rationale or "")[:160]
        ticks_away = effect.fire_tick - world.tick
        _add(
            QuestJournalEntry(
                entry_id=f"sched:{effect.effect_id}",
                source="scheduled",
                title=f"Something brewing (~{ticks_away} ticks)",
                detail=detail,
                priority=0.45,
                tick=effect.fire_tick,
                status="upcoming",
            )
        )

    entries.sort(key=lambda e: (-e.priority, e.tick))
    return entries[:max_entries]


def format_journal_lines(entries: list[QuestJournalEntry]) -> list[str]:
    """Compact strings for REPL / autonomous output."""
    lines: list[str] = []
    for e in entries:
        prefix = {
            "pack": "Quest",
            "emergent": "Concern",
            "npc": "Nearby",
            "scheduled": "Soon",
            "rumour": "Rumour",
        }.get(e.source, "Note")
        status = f" [{e.status}]" if e.status != "active" else ""
        lines.append(f"{prefix}{status}: {e.title}")
        if e.detail and e.detail != e.title:
            lines.append(f"  → {e.detail[:120]}")
    return lines


__all__ = [
    "QuestJournalEntry",
    "build_quest_journal",
    "format_journal_lines",
]
