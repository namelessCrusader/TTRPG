"""
Versioned run traces for research experiments.

Trace format version: 1
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .schemas import SemanticAction, ValidationResult, WorldState
from .state_hash import world_state_fingerprint


TRACE_VERSION = 1


@dataclass
class TraceRecorder:
    """Append-only JSONL trace writer attached to a GameLoop run."""

    run_id: str
    world_pack: str
    seed: int
    adapter_label: str
    policy_label: str
    output_path: Optional[Path] = None
    _fh: Any = field(default=None, repr=False)
    _tick_records: list[dict[str, Any]] = field(default_factory=list)
    _started_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.output_path, "w", encoding="utf-8")
            self._write(
                {
                    "type": "run_meta",
                    "trace_version": TRACE_VERSION,
                    "run_id": self.run_id,
                    "world_pack": self.world_pack,
                    "seed": self.seed,
                    "adapter": self.adapter_label,
                    "policy": self.policy_label,
                }
            )

    def close(self, world: WorldState, *, goals_completed: int = 0) -> dict[str, Any]:
        summary = {
            "type": "run_summary",
            "ticks": world.tick,
            "final_state_hash": world_state_fingerprint(world),
            "events": len(world.event_log),
            "goals_completed": goals_completed,
            "elapsed_s": round(time.time() - self._started_at, 3),
        }
        self._write(summary)
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        return summary

    def record_entity_step(
        self,
        *,
        tick: int,
        entity_id: str,
        entity_name: str,
        action: SemanticAction,
        validation: ValidationResult,
        policy_branch: str,
        projection_tokens: int = 0,
    ) -> None:
        record = {
            "type": "entity_step",
            "tick": tick,
            "entity_id": str(entity_id),
            "entity_name": entity_name,
            "verb": str(action.verb),
            "target": (
                str(action.target)
                if isinstance(action.target, str)
                else (
                    {"x": action.target.x, "y": action.target.y}
                    if action.target is not None and hasattr(action.target, "x")
                    else None
                )
            ),
            "valid": validation.valid,
            "rejection": (
                validation.rejection_reason.value
                if validation.rejection_reason and hasattr(validation.rejection_reason, "value")
                else str(validation.rejection_reason) if validation.rejection_reason else None
            ),
            "transitions": [
                t.kind.value for t in (validation.concrete_transitions or [])
            ],
            "policy_branch": policy_branch,
            "projection_tokens": projection_tokens,
        }
        self._tick_records.append(record)
        self._write(record)

    def _write(self, obj: dict[str, Any]) -> None:
        if self._fh is not None:
            self._fh.write(json.dumps(obj, default=str) + "\n")
            self._fh.flush()


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:10]}"
