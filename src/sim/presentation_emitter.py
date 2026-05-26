"""
Build presentation-safe TickFrames from engine results.

Clients (REPL, pygame, web, Unity) consume TickFrame — never raw WorldState.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from .presentation_contract import (
    EntityActionView,
    PresentationStream,
    TickFrame,
    TransitionView,
)
from .schemas import EntityId

if TYPE_CHECKING:
    from .game_loop import AutonomousTickResult, StepResult
    from .schemas import Event, WorldState


def _entity_name(world: "WorldState", eid: str) -> str:
    ent = world.spatial.entities.get(EntityId(eid))
    if ent:
        return ent.name
    node = world.relational.nodes.get(eid)
    return node.name if node else eid


def _transitions_for_event(event: Optional["Event"]) -> list[TransitionView]:
    if event is None:
        return []
    return [
        TransitionView(kind=t.kind.value, payload=dict(t.payload))
        for t in event.transitions
    ]


def _action_view(
    result: "StepResult",
    world: "WorldState",
) -> EntityActionView:
    from .narrator import render_event

    actor_id = str(result.action.actor)
    narration = result.narration or ""
    if not narration and result.event:
        narration = render_event(result.event, world)
    return EntityActionView(
        tick=result.tick,
        entity_id=actor_id,
        entity_name=_entity_name(world, actor_id),
        verb=str(result.action.verb),
        narration=narration,
        transitions=_transitions_for_event(result.event),
    )


def build_tick_frame(
    result: "StepResult",
    world: "WorldState",
    *,
    player_id: Optional[EntityId] = None,
) -> TickFrame:
    """Build one TickFrame from a player step (includes NPC sub-actions)."""
    actions = [_action_view(result, world)]
    for nr in result.npc_results:
        actions.append(_action_view(nr, world))

    ambient = [
        ev.narrative_hint or "(ambient)"
        for ev in result.ambient_events
        if ev.narrative_hint or ev.transitions
    ]
    for line in result.pressure_lines or []:
        ambient.append(line)

    chronicle: list[str] = []
    if player_id is not None:
        from .episodic_memory import chronicle_lines

        chronicle = chronicle_lines(world, max_entries=3)

    facts = [f.claim[:120] for f in world.world_facts[-4:]]

    return TickFrame(
        tick=result.tick,
        projection=result.projection,
        entity_actions=actions,
        ambient_narration=ambient,
        chronicle=chronicle,
        world_facts=facts,
        completed_goals=[
            g.title for g in result.completed_goals if hasattr(g, "title")
        ],
        quest_journal=[
            e.title if hasattr(e, "title") else str(e)
            for e in getattr(result, "quest_journal", []) or []
        ],
    )


def build_tick_frame_autonomous(
    result: "AutonomousTickResult",
    world: "WorldState",
    projection: Optional[object] = None,
) -> TickFrame:
    """Build TickFrame from a headless autonomous tick."""
    from .schemas import SemanticProjection

    actions: list[EntityActionView] = []
    for er in result.entity_results:
        actions.append(_action_view(er, world))

    ambient = [
        ev.narrative_hint or "(ambient)"
        for ev in result.ambient_events
        if ev.narrative_hint or ev.transitions
    ]
    ambient.extend(result.physics_lines or [])
    ambient.extend(result.semantic_lines or [])
    ambient.extend(result.pressure_lines or [])
    ambient.extend(result.scenario_lines or [])

    proj = (
        projection
        if isinstance(projection, SemanticProjection)
        else SemanticProjection(tick=result.tick, focal_entity=EntityId("system"))
    )

    return TickFrame(
        tick=result.tick,
        projection=proj,
        entity_actions=actions,
        ambient_narration=ambient,
        world_facts=[f.claim[:120] for f in world.world_facts[-4:]],
        quest_journal=[
            e.title if hasattr(e, "title") else str(e)
            for e in getattr(result, "quest_journal", []) or []
        ],
    )


class PresentationCollector:
    """
    Accumulates TickFrames for streaming export (SSE / WebSocket / replay files).
    """

    def __init__(
        self,
        world_name: str,
        *,
        run_id: Optional[str] = None,
        max_frames: int = 500,
    ) -> None:
        self.stream = PresentationStream(
            world_name=world_name,
            run_id=run_id or f"run_{uuid.uuid4().hex[:8]}",
        )
        self.max_frames = max_frames

    def push(self, frame: TickFrame) -> None:
        self.stream.frames.append(frame)
        if len(self.stream.frames) > self.max_frames:
            self.stream.frames = self.stream.frames[-self.max_frames :]

    @property
    def last_frame(self) -> Optional[TickFrame]:
        return self.stream.frames[-1] if self.stream.frames else None

    def to_json(self) -> str:
        return self.stream.to_json()

    def to_jsonl_line(self, frame: TickFrame) -> str:
        return frame.model_dump_json()
