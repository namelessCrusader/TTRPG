"""
Presentation-layer contract: what clients receive (never full WorldState).

Used by future Unity/web front-ends and trace stream consumers.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from .schemas import SemanticProjection


class TransitionView(BaseModel):
    kind: str
    payload: dict[str, Any] = Field(default_factory=dict)


class EntityActionView(BaseModel):
    tick: int
    entity_id: str
    entity_name: str
    verb: str
    narration: str = ""
    transitions: list[TransitionView] = Field(default_factory=list)


class TickFrame(BaseModel):
    """One tick of presentation-safe data."""

    tick: int
    projection: SemanticProjection
    entity_actions: list[EntityActionView] = Field(default_factory=list)
    ambient_narration: list[str] = Field(default_factory=list)
    chronicle: list[str] = Field(default_factory=list)
    world_facts: list[str] = Field(default_factory=list)
    completed_goals: list[str] = Field(default_factory=list)
    quest_journal: list[str] = Field(default_factory=list)
    lm_metrics: dict[str, Any] = Field(default_factory=dict)


class PresentationStream(BaseModel):
    """
    Stream envelope for WebSocket / SSE clients.

    Engine emits TickFrame; client renders animation + UI.
    """

    world_name: str
    run_id: str = ""
    frames: list[TickFrame] = Field(default_factory=list)

    def to_json(self) -> str:
        return self.model_dump_json()
