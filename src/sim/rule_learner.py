"""
Rule-learner: capture successful LM adjudications as candidate pack rules.

When the LM adjudicator successfully rescues a creative player action, the
resulting facts and transition proposals are a one-off solution that does
not compound — the next time a similar action happens, the LM has to
improvise again, possibly differently.  That is the difference between
"DM rules-of-thumb" and "simulation".

This module bridges the gap by serialising successful adjudications as
candidate ``VerbTemplate``-shaped YAML entries under
``<pack_dir>/learned/<date>.yaml``.  A human (or a second LM pass with
a strict schema) can later promote vetted entries into the canonical
``verb_templates.yaml`` / ``open_verbs.yaml`` / ``property_interactions.yaml``
files, at which point the engine handles that intent deterministically
forever after.

Design constraints
------------------

- **Opt-in only.**  Reads ``world.config.extra["learn_from_adjudication"]``
  (default ``False``).  Tests and headless smoke runs do not write files
  unless the pack explicitly enables learning.
- **Idempotent.**  Dedupes by a stable hash of (verb, sorted proposal
  kinds, fact tags) so the same rescued intent is not recorded twice in
  the same session.
- **Never auto-loaded.**  The engine ships no loader for ``learned/`` —
  promotion to the canonical pack is an explicit human/curator step.
- **Safe to call.**  Any failure (missing pack path, unwritable dir,
  serialisation error) logs a warning and returns without raising.
- **Schema-stable.**  Output mirrors ``verb_templates.yaml`` so a curator
  can paste an entry directly into the canonical file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .schemas import (
        AdjudicationResult,
        SemanticAction,
        WorldState,
    )

logger = logging.getLogger(__name__)

# Per-process dedupe set — same (world, action shape) only recorded once
# per run.  Cleared automatically on process exit.
_seen_hashes: set[str] = set()
_seen_lock = threading.Lock()

# Marker key inside ``world.config.extra``.
_ENABLE_KEY: str = "learn_from_adjudication"

# Folder name written under the pack directory.
_LEARNED_DIR: str = "learned"


def is_enabled(world: "WorldState") -> bool:
    """True when the world pack opts into rule learning."""
    try:
        return bool(world.config.extra.get(_ENABLE_KEY, False))
    except Exception:
        return False


def _pack_path(world: "WorldState") -> Optional[Path]:
    raw = (world.meta or {}).get("_pack_path")
    if not raw:
        return None
    try:
        return Path(str(raw))
    except Exception:
        return None


def _stable_hash(
    verb: str,
    proposal_kinds: list[str],
    fact_tags: list[str],
) -> str:
    """Hash that ignores ordering inside lists and ignores incidental detail."""
    payload = json.dumps(
        {
            "verb": verb.lower().strip(),
            "kinds": sorted(set(proposal_kinds)),
            "tags": sorted(set(fact_tags)),
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _serialise_payload(payload: Any) -> Any:
    """Convert payload values to YAML-friendly primitives."""
    if isinstance(payload, dict):
        return {str(k): _serialise_payload(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_serialise_payload(v) for v in payload]
    if payload is None or isinstance(payload, (str, int, float, bool)):
        return payload
    return str(payload)


def _build_yaml_entry(
    *,
    rule_id: str,
    verb: str,
    intent: str,
    tick: int,
    actor_id: str,
    actor_role: str,
    ruling_text: str,
    proposal_kinds: list[str],
    proposals_serialised: list[dict[str, Any]],
    fact_tags: list[str],
    synthesized_verb: Optional[str],
) -> str:
    """
    Render a single learned rule as a YAML string ready to paste into
    ``verb_templates.yaml``.

    We intentionally write YAML by hand (rather than via PyYAML's dumper)
    so the output is stable, comment-rich, and reviewable in a code review.
    """
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _fmt_payload(payload: dict[str, Any]) -> list[str]:
        if not payload:
            return ["        payload: {}"]
        lines = ["        payload:"]
        for k, v in payload.items():
            if isinstance(v, (dict, list)):
                lines.append(
                    f"          {k}: "
                    f"{json.dumps(v, ensure_ascii=True)}"
                )
            elif isinstance(v, str):
                # Quote strings; escape any embedded quotes / newlines.
                safe = v.replace("\\", "\\\\").replace('"', '\\"')
                safe = safe.replace("\n", " ").replace("\r", " ")
                lines.append(f'          {k}: "{safe}"')
            else:
                lines.append(f"          {k}: {json.dumps(v)}")
        return lines

    out: list[str] = []
    out.append(f"# ── learned {timestamp} ──")
    out.append(f"#   rule_id      : {rule_id}")
    out.append(f"#   world_tick   : {tick}")
    out.append(f"#   actor        : {actor_id} ({actor_role or 'unknown role'})")
    out.append(f"#   original     : {intent[:200]!r}")
    if synthesized_verb and synthesized_verb != verb:
        out.append(f"#   synthesized_verb (DM): {synthesized_verb}")
    if ruling_text:
        ruling_one_line = " ".join(ruling_text.split())[:200]
        out.append(f"#   ruling       : {ruling_one_line}")
    if fact_tags:
        out.append(f"#   fact_tags    : {sorted(set(fact_tags))}")
    out.append(f"  - verb: {verb}")
    out.append("    on_success:")
    if proposals_serialised:
        for prop in proposals_serialised:
            kind = prop.get("kind") or "unknown"
            out.append(f"      - kind: {kind}")
            out.extend(_fmt_payload(prop.get("payload") or {}))
    else:
        out.append("      []  # adjudication produced facts only; review manually")
    out.append("")  # trailing blank line
    return "\n".join(out)


def record_adjudication(
    world: "WorldState",
    action: "SemanticAction",
    intent: str,
    adj: "AdjudicationResult",
    *,
    file_basename: Optional[str] = None,
) -> Optional[Path]:
    """
    Append a learned-rule candidate to ``<pack>/learned/<date>.yaml``.

    Returns the path written to, or ``None`` when learning is disabled
    or no useful content was present.  All errors are caught and logged
    — this function never raises.
    """
    if not is_enabled(world):
        return None

    proposals = list(getattr(adj, "transition_proposals", None) or [])
    facts = list(getattr(adj, "facts", None) or [])
    if not proposals and not facts and not getattr(adj, "synthesized_verb", None):
        # Nothing to learn from — the LM only produced ruling prose.
        return None

    pack_dir = _pack_path(world)
    if pack_dir is None:
        logger.debug("rule_learner: pack path missing on world.meta; skipped.")
        return None

    verb = str(getattr(action, "verb", "") or "unknown").lower().strip()
    proposal_kinds = [str(p.kind) for p in proposals]
    fact_tags: list[str] = []
    for f in facts:
        fact_tags.extend(getattr(f, "tags", None) or [])

    rule_hash = _stable_hash(verb, proposal_kinds, fact_tags)
    with _seen_lock:
        if rule_hash in _seen_hashes:
            return None
        _seen_hashes.add(rule_hash)

    rule_id = f"learned_{verb}_{rule_hash}"

    actor_id = str(getattr(action, "actor", "")) or "unknown_actor"
    actor_ent = None
    try:
        actor_ent = world.spatial.entities.get(action.actor)
    except Exception:
        actor_ent = None
    actor_role = ""
    if actor_ent is not None:
        actor_role = getattr(actor_ent, "role", "") or ""

    proposals_serialised: list[dict[str, Any]] = []
    for prop in proposals:
        proposals_serialised.append(
            {
                "kind": str(prop.kind),
                "payload": _serialise_payload(dict(prop.payload or {})),
            }
        )

    entry = _build_yaml_entry(
        rule_id=rule_id,
        verb=verb,
        intent=(intent or getattr(action, "raw_input", "") or "")[:240],
        tick=int(getattr(world, "tick", 0) or 0),
        actor_id=actor_id,
        actor_role=actor_role,
        ruling_text=str(getattr(adj, "ruling_text", "") or ""),
        proposal_kinds=proposal_kinds,
        proposals_serialised=proposals_serialised,
        fact_tags=fact_tags,
        synthesized_verb=getattr(adj, "synthesized_verb", None),
    )

    try:
        learned_dir = pack_dir / _LEARNED_DIR
        learned_dir.mkdir(parents=True, exist_ok=True)
        fname = file_basename or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        target = learned_dir / f"{fname}.yaml"
        header_needed = not target.exists()
        with target.open("a", encoding="utf-8") as fh:
            if header_needed:
                fh.write(
                    "# Auto-recorded LM adjudication candidates.\n"
                    "# Schema mirrors verb_templates.yaml — paste promoted entries\n"
                    "# under the canonical `templates:` mapping after review.\n"
                    "# This file is NEVER auto-loaded by the engine.\n"
                    "templates:\n"
                )
            fh.write(entry)
        logger.info(
            "rule_learner: recorded %s → %s", rule_id, target,
        )
        return target
    except Exception as exc:
        logger.warning("rule_learner: failed to write learned rule: %s", exc)
        return None


def reset_dedupe() -> None:
    """Test helper: clear the per-process dedupe set."""
    with _seen_lock:
        _seen_hashes.clear()


__all__ = [
    "is_enabled",
    "record_adjudication",
    "reset_dedupe",
]
