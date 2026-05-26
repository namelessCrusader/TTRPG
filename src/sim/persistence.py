"""
World state persistence.

save_world()       — serialize WorldState to JSON
load_world()       — deserialize with version migration
save_checkpoint()  — rotating checkpoint slots
verify_save_replay() — replay integrity check
"""

from __future__ import annotations

import copy
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from .schemas import WorldState

logger = logging.getLogger(__name__)

SAVEGAME_VERSION = "3"
_SUPPORTED_VERSIONS = frozenset({"1", "2", "3"})
_DEFAULT_CHECKPOINT_SLOTS = 5


def migrate_save_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize a savegame wrapper to the current schema version.

    Handles legacy saves missing Phase 1/2/3 fields by applying defaults
    before Pydantic validation.
    """
    version = str(raw.get("version", "1"))
    if version not in _SUPPORTED_VERSIONS:
        logger.warning("Unknown savegame version %r — attempting load anyway", version)

    state = raw.get("state", raw)
    if not isinstance(state, dict):
        raise ValueError("Savegame missing 'state' object")

    # Phase 1 adjudication fields
    state.setdefault("world_facts", [])
    state.setdefault("scheduled_effects", [])
    # Phase 2 chronicle
    state.setdefault("player_chronicle", [])
    state.setdefault("world_chronicle", [])
    # Ensure meta dict exists
    meta = state.setdefault("meta", {})
    if isinstance(meta, dict):
        meta.setdefault("pack_id", raw.get("pack_id"))

    raw["version"] = SAVEGAME_VERSION
    raw["state"] = state
    return raw


def save_world(
    world: WorldState,
    path: Union[str, Path],
    *,
    pack_id: Optional[str] = None,
    max_events: Optional[int] = None,
    lm_metrics: Optional[dict[str, Any]] = None,
) -> int:
    """
    Serialize ``world`` to a JSON savegame file.

    Parameters
    ----------
    pack_id    : Optional pack directory name for migration hints.
    max_events : If set, only the last N events are persisted.
    lm_metrics : Optional LM usage snapshot to embed in the save wrapper.
    """
    p = Path(path)
    if not p.suffix:
        p = p.with_suffix(".json")

    event_log = world.event_log
    if max_events is not None and max_events > 0 and len(event_log) > max_events:
        event_log = event_log[-max_events:]

    state = json.loads(world.model_dump_json())
    state["event_log"] = [json.loads(e.model_dump_json()) for e in event_log]

    wrapper = {
        "version": SAVEGAME_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "saved_at_tick": world.tick,
        "world_name": world.name,
        "pack_id": pack_id or world.meta.get("pack_id"),
        "event_count": len(event_log),
        "lm_metrics": lm_metrics or world.meta.get("lm_metrics"),
        "state": state,
    }
    p.write_text(json.dumps(wrapper, indent=2), encoding="utf-8")
    logger.info(
        "Saved world '%s' to %s (tick=%d, events=%d)",
        world.name,
        p,
        world.tick,
        len(event_log),
    )
    return len(event_log)


def load_world(
    path: Union[str, Path],
    *,
    pack_path: Optional[Union[str, Path]] = None,
    rebind_config: bool = False,
) -> WorldState:
    """
    Deserialize a WorldState from a savegame file.

    Parameters
    ----------
    pack_path     : Path to world pack for optional config rebind.
    rebind_config : When True and ``pack_path`` is set, refresh ``world.config``
                    from the live pack (entities/state from save are kept).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Savegame not found: {p}")

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Savegame is not valid JSON: {exc}") from exc

    raw = migrate_save_dict(raw)
    state_dict = raw.get("state", raw)
    try:
        world = WorldState.model_validate(state_dict)
    except Exception as exc:
        raise ValueError(f"Could not reconstruct WorldState: {exc}") from exc

    if raw.get("pack_id"):
        world.meta["pack_id"] = raw["pack_id"]
    world.meta["savegame_version"] = raw.get("version", SAVEGAME_VERSION)
    if raw.get("lm_metrics"):
        world.meta["lm_metrics"] = raw["lm_metrics"]

    if rebind_config and pack_path is not None:
        from .world_loader import load_world_pack

        fresh = load_world_pack(str(pack_path))
        world.config = fresh.config
        world.meta["pack_rebound_at"] = world.tick
        logger.info(
            "Rebound world.config from pack %s at tick %d",
            pack_path,
            world.tick,
        )

    logger.info(
        "Loaded world '%s' from %s (tick=%d, events=%d)",
        world.name,
        p,
        world.tick,
        len(world.event_log),
    )
    return world


def save_checkpoint(
    world: WorldState,
    directory: Union[str, Path],
    *,
    pack_id: Optional[str] = None,
    max_slots: int = _DEFAULT_CHECKPOINT_SLOTS,
    lm_metrics: Optional[dict[str, Any]] = None,
) -> Path:
    """
    Save a rotating checkpoint (slot 01 = newest).

    Returns path to the written checkpoint file.
    """
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    slots = max(1, min(max_slots, 20))

    oldest = d / f"checkpoint_{slots:02d}.json"
    if oldest.exists():
        oldest.unlink()
    for i in range(slots - 1, 0, -1):
        src = d / f"checkpoint_{i:02d}.json"
        dst = d / f"checkpoint_{i + 1:02d}.json"
        if src.exists():
            shutil.move(str(src), str(dst))

    target = d / "checkpoint_01.json"
    save_world(
        world,
        target,
        pack_id=pack_id,
        lm_metrics=lm_metrics,
    )

    latest = d / "latest.json"
    latest.write_text(
        json.dumps(
            {
                "path": target.name,
                "tick": world.tick,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return target


def list_checkpoints(directory: Union[str, Path]) -> list[Path]:
    """Return checkpoint files newest-first (slot 01 is always most recent)."""
    d = Path(directory)
    if not d.is_dir():
        return []
    one = d / "checkpoint_01.json"
    rest = sorted(
        p for p in d.glob("checkpoint_*.json") if p.name != "checkpoint_01.json"
    )
    rest.reverse()
    if one.exists():
        return [one, *rest]
    return rest


def load_latest_checkpoint(
    directory: Union[str, Path],
    **load_kwargs: Any,
) -> WorldState:
    """Load newest checkpoint (always slot 01)."""
    d = Path(directory)
    target = d / "checkpoint_01.json"
    if not target.exists():
        checkpoints = list_checkpoints(directory)
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints in {directory}")
        target = checkpoints[0]
    return load_world(target, **load_kwargs)


def verify_save_replay(
    world: WorldState,
    *,
    initial_world: Optional[WorldState] = None,
) -> tuple[bool, str]:
    """
    Replay ``world.event_log`` from a tick-0 snapshot and compare fingerprints.

    ``initial_world`` must be a deep copy of the world *before* the logged
    events were applied (same pattern as ``test_state_hash_replay``).
    """
    from .game_loop import GameLoop
    from .lm_adapter import MockLMAdapter
    from .state_hash import world_state_fingerprint

    if not world.event_log:
        return True, "no events to replay"
    if initial_world is None:
        return True, "skipped: no tick-0 snapshot supplied"

    baseline_fp = world_state_fingerprint(world)
    replayed = GameLoop(world, adapter=MockLMAdapter()).replay(
        copy.deepcopy(initial_world)
    )

    replay_fp = world_state_fingerprint(replayed)
    if replay_fp == baseline_fp:
        return True, f"fingerprint match ({replay_fp})"
    return False, f"fingerprint mismatch: live={baseline_fp} replay={replay_fp}"
