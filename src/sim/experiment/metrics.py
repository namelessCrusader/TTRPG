"""
Aggregate metrics from trace JSONL files (no LM in scorer).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunMetrics:
    run_id: str = ""
    world_pack: str = ""
    seed: int = 0
    adapter: str = ""
    policy: str = ""
    ticks: int = 0
    events: int = 0
    entity_steps: int = 0
    valid_rate: float = 0.0
    speak_fraction: float = 0.0
    unique_verbs: int = 0
    repeat_bigram_rate: float = 0.0
    policy_branches: dict[str, int] = field(default_factory=dict)
    final_state_hash: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "world_pack": self.world_pack,
            "seed": self.seed,
            "adapter": self.adapter,
            "policy": self.policy,
            "ticks": self.ticks,
            "events": self.events,
            "entity_steps": self.entity_steps,
            "valid_rate": round(self.valid_rate, 4),
            "speak_fraction": round(self.speak_fraction, 4),
            "unique_verbs": self.unique_verbs,
            "repeat_bigram_rate": round(self.repeat_bigram_rate, 4),
            "final_state_hash": self.final_state_hash,
            **{f"branch_{k}": v for k, v in self.policy_branches.items()},
        }


def score_run_trace(path: Path) -> RunMetrics:
    """Parse a trace JSONL file and compute aggregates."""
    m = RunMetrics()
    steps: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("type") == "run_meta":
            m.run_id = rec.get("run_id", "")
            m.world_pack = rec.get("world_pack", "")
            m.seed = int(rec.get("seed", 0))
            m.adapter = rec.get("adapter", "")
            m.policy = rec.get("policy", "")
        elif rec.get("type") == "run_summary":
            m.ticks = int(rec.get("ticks", 0))
            m.events = int(rec.get("events", 0))
            m.final_state_hash = rec.get("final_state_hash", "")
        elif rec.get("type") == "entity_step":
            steps.append(rec)

    m.entity_steps = len(steps)
    if not steps:
        return m

    valid = sum(1 for s in steps if s.get("valid"))
    m.valid_rate = valid / len(steps)
    verbs = [str(s.get("verb", "")).lower() for s in steps]
    m.unique_verbs = len(set(verbs))
    speakish = sum(
        1 for v in verbs
        if v in ("speak", "ask", "say", "tell", "whisper", "shout", "sing")
    )
    m.speak_fraction = speakish / len(steps)
    branches = Counter(str(s.get("policy_branch", "unknown")) for s in steps)
    m.policy_branches = dict(branches)

    # Repetition: duplicate consecutive verb+target pairs
    pairs = [
        (verbs[i], str(steps[i].get("target", "")))
        for i in range(len(steps))
    ]
    repeats = sum(1 for i in range(1, len(pairs)) if pairs[i] == pairs[i - 1])
    m.repeat_bigram_rate = repeats / max(1, len(pairs) - 1)
    return m


def write_csv_report(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for row in rows:
        lines.append(",".join(_csv_escape(row.get(k, "")) for k in keys))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _csv_escape(val: Any) -> str:
    s = str(val)
    if re.search(r'[,"\n]', s):
        return '"' + s.replace('"', '""') + '"'
    return s
