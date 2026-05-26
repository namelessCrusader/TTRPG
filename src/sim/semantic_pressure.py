"""
Semantic pressure adapters and tick-phase orchestration.

Mock adapter provides deterministic interpretation for tests and CI without GPU.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .lm_adapter import LMAdapter

from .schemas import (
    SemanticDelta,
    SemanticDynamicsConfig,
    SemanticLensKind,
    SemanticSlice,
    WorldState,
)
from .semantic_compiler import (
    SemanticCompileResult,
    apply_semantic_transitions,
    compile_deltas,
)
from .semantic_slice import (
    build_slice,
    clear_lens_buffer,
    ensure_semantic_state,
    ingest_new_events,
    lenses_with_pending_buffers,
    record_tick_events,
    should_run_semantic_pass,
)

logger = logging.getLogger(__name__)

_VIOLENCE_VERBS = frozenset({"attack", "accuse", "threaten", "intimidate", "fight"})


class SemanticPressureAdapter(ABC):
    @abstractmethod
    def infer_pressure(self, slice_: SemanticSlice) -> list[SemanticDelta]:
        ...


class MockSemanticPressureAdapter(SemanticPressureAdapter):
    """
    Rule-based semantic interpretation for development and tests.

    Reacts to violence, accusations, and deaths in the slice with edge shifts,
    beliefs, faction narratives, and emergent goals.
    """

    def infer_pressure(self, slice_: SemanticSlice) -> list[SemanticDelta]:
        deltas: list[SemanticDelta] = []
        violent = [
            e for e in slice_.canonical_events
            if e.verb in _VIOLENCE_VERBS
            or "entity_died" in e.transition_kinds
            or (
                "entity_health_changed" in e.transition_kinds
                and "attack" in e.summary.lower()
            )
        ]
        if not violent:
            # Low-key social tick: one mild belief drift if dialogue occurred
            for ev in slice_.canonical_events:
                if ev.verb in ("speak", "gossip", "accuse", "whisper"):
                    if slice_.lens == SemanticLensKind.AGENT and slice_.lens_id == ev.actor_id:
                        continue
                    if slice_.lens == SemanticLensKind.AGENT:
                        deltas.append(
                            SemanticDelta(
                                kind="belief_mutated",
                                rationale_event_id=ev.event_id,
                                source=slice_.lens_id,
                                target=ev.actor_id,
                                claim=f"Heard that {ev.summary}",
                                fidelity=0.45,
                            )
                        )
                        break
            return deltas[:3]

        primary = violent[0]
        evt_id = primary.event_id

        if slice_.lens == SemanticLensKind.FACTION:
            theme = "moral_panic" if len(violent) >= 2 else "legitimacy_crisis"
            deltas.append(
                SemanticDelta(
                    kind="faction_narrative",
                    rationale_event_id=evt_id,
                    source=slice_.lens_id,
                    theme=theme,
                    intensity=min(1.0, 0.35 + 0.15 * len(violent)),
                )
            )
            if len(violent) >= 2:
                deltas.append(
                    SemanticDelta(
                        kind="taboo",
                        rationale_event_id=evt_id,
                        source=slice_.lens_id,
                        rule="open violence in shared spaces",
                        strength=0.55,
                    )
                )
            deltas.append(
                SemanticDelta(
                    kind="social_momentum",
                    rationale_event_id=evt_id,
                    scope=f"faction:{slice_.lens_id}",
                    axis="violence_vs_reconciliation",
                    delta=0.12 * len(violent),
                )
            )
            return deltas

        if slice_.lens == SemanticLensKind.AGENT:
            actor = primary.actor_id
            target = primary.target_id
            lid = slice_.lens_id

            if target and lid in (actor, target):
                other = target if lid == actor else actor
                deltas.append(
                    SemanticDelta(
                        kind="edge_shift",
                        rationale_event_id=evt_id,
                        source=lid,
                        target=other,
                        edge_kind="distrusts",
                        delta=0.15,
                    )
                )
                deltas.append(
                    SemanticDelta(
                        kind="emotional_contagion",
                        rationale_event_id=evt_id,
                        source=lid,
                        theme="fear",
                        intensity=0.4,
                    )
                )
            elif lid not in (actor, target or ""):
                deltas.append(
                    SemanticDelta(
                        kind="belief_mutated",
                        rationale_event_id=evt_id,
                        source=lid,
                        target=actor,
                        claim=f"Witnessed: {primary.summary}",
                        fidelity=0.65,
                    )
                )
                if "entity_died" in primary.transition_kinds:
                    deltas.append(
                        SemanticDelta(
                            kind="goal_emerged",
                            rationale_event_id=evt_id,
                            source=lid,
                            goal="Understand what happened and who is responsible",
                            priority=0.75,
                        )
                    )
            else:
                deltas.append(
                    SemanticDelta(
                        kind="suspicion_focus",
                        rationale_event_id=evt_id,
                        source=lid,
                        claim=primary.summary[:80],
                    )
                )

            if len(violent) >= 2:
                deltas.append(
                    SemanticDelta(
                        kind="social_momentum",
                        rationale_event_id=evt_id,
                        scope="local",
                        axis="violence_vs_reconciliation",
                        delta=0.1,
                    )
                )
            return deltas

        return deltas


@dataclass
class SemanticPassResult:
    ran: bool = False
    trigger: str = ""
    lenses_processed: int = 0
    deltas_applied: int = 0
    log_lines: list[str] = field(default_factory=list)


_ALLOWED_DELTA_KINDS = frozenset({
    "edge_shift",
    "belief_mutated",
    "goal_emerged",
    "faction_narrative",
    "social_momentum",
    "taboo",
    "suspicion_focus",
    "emotional_contagion",
})


def build_semantic_pressure_prompts(slice_: SemanticSlice) -> tuple[str, str]:
    """System + user prompts for LM semantic interpretation."""
    events_block = "\n".join(
        f"- [{ev.event_id}] tick={ev.tick} {ev.actor_name} {ev.verb}"
        + (f" → {ev.target_name}" if ev.target_name else "")
        + f": {ev.summary}"
        for ev in slice_.canonical_events
    )
    system = (
        "You interpret recent canonical simulation events for one lens "
        f"({slice_.lens.value}: {slice_.lens_name}). "
        "Propose bounded semantic consequences — beliefs, distrust shifts, "
        "emergent goals, faction narratives — grounded ONLY in listed events. "
        "Each delta MUST cite rationale_event_id from the list. "
        "Return JSON: {\"deltas\": [...]}. "
        f"Allowed kinds: {', '.join(sorted(_ALLOWED_DELTA_KINDS))}. "
        "Max 5 deltas."
    )
    user = (
        f"Lens: {slice_.lens.value} / {slice_.lens_id}\n"
        f"Trigger: {slice_.trigger.value}\n"
        f"Ticks {slice_.tick_from}–{slice_.tick_to}\n\n"
        f"Canonical events:\n{events_block or '(none)'}\n\n"
        f"Prior beliefs:\n"
        + ("\n".join(f"- {b}" for b in slice_.prior_beliefs[:6]) or "(none)")
        + "\n\nPropose semantic deltas as JSON."
    )
    return system, user


def parse_semantic_deltas(raw: str, *, max_deltas: int = 8) -> list[SemanticDelta]:
    """Parse LM JSON into validated SemanticDelta objects."""
    import json as _json
    import re as _re

    text = raw.strip()
    if not text:
        return []
    fence = _re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = _json.loads(text)
    except _json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return []
        data = _json.loads(text[start : end + 1])

    items = data.get("deltas") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []

    deltas: list[SemanticDelta] = []
    for item in items[:max_deltas]:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", "")).strip()
        evt_id = str(item.get("rationale_event_id", "")).strip()
        if kind not in _ALLOWED_DELTA_KINDS or not evt_id:
            continue
        try:
            deltas.append(SemanticDelta.model_validate(item))
        except Exception:
            logger.debug("semantic: skip invalid delta %r", item)
    return deltas


SEMANTIC_DELTA_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "deltas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "rationale_event_id": {"type": "string"},
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "edge_kind": {"type": "string"},
                    "delta": {"type": "number"},
                    "weight": {"type": "number"},
                    "claim": {"type": "string"},
                    "fidelity": {"type": "number"},
                    "goal": {"type": "string"},
                    "priority": {"type": "number"},
                    "theme": {"type": "string"},
                    "intensity": {"type": "number"},
                    "scope": {"type": "string"},
                    "axis": {"type": "string"},
                    "rule": {"type": "string"},
                    "strength": {"type": "number"},
                },
                "required": ["kind", "rationale_event_id"],
            },
        }
    },
    "required": ["deltas"],
}


class LMSemanticPressureAdapter(SemanticPressureAdapter):
    """
    LM-driven semantic interpretation with rule-based fallback.

    Used when ``world.config.semantic.use_mock`` is False and a real
    ``LMAdapter`` is available on the game loop.
    """

    def __init__(
        self,
        lm_adapter: "LMAdapter",
        *,
        fallback: Optional[SemanticPressureAdapter] = None,
        max_deltas: int = 8,
    ) -> None:
        self._lm = lm_adapter
        self._fallback = fallback or MockSemanticPressureAdapter()
        self._max_deltas = max_deltas

    def infer_pressure(self, slice_: SemanticSlice) -> list[SemanticDelta]:
        try:
            deltas = self._lm.infer_semantic_pressure(slice_)
            if deltas:
                return deltas[: self._max_deltas]
        except NotImplementedError:
            logger.debug(
                "semantic: %s lacks infer_semantic_pressure — rule fallback",
                type(self._lm).__name__,
            )
        except Exception as exc:
            logger.warning("semantic LM infer failed: %s — rule fallback", exc)
        return self._fallback.infer_pressure(slice_)


def get_semantic_adapter(
    world: WorldState,
    lm_adapter: Optional["LMAdapter"] = None,
) -> SemanticPressureAdapter:
    """
    Resolve the semantic pressure adapter for this world.

    - ``use_mock=True`` (default): deterministic rule-based interpreter.
    - ``use_mock=False`` + ``lm_adapter``: LM with rule-based fallback.
    - ``use_mock=False`` without LM: rule-based with a warning.
    """
    cfg = world.config.semantic
    if cfg.use_mock:
        return MockSemanticPressureAdapter()
    if lm_adapter is not None:
        return LMSemanticPressureAdapter(
            lm_adapter,
            max_deltas=cfg.max_deltas_per_lens,
        )
    logger.warning(
        "semantic.use_mock=false but no LM adapter wired — using rule-based interpreter"
    )
    return MockSemanticPressureAdapter()


def run_semantic_dynamics_pass(
    world: WorldState,
    tick: int,
    *,
    adapter: Optional[SemanticPressureAdapter] = None,
    lm_adapter: Optional["LMAdapter"] = None,
    force: bool = False,
) -> SemanticPassResult:
    """
    Run delayed semantic interpretation if scheduled or forced.

    Call after physical ticks and deterministic propagation; typically after
    ``record_tick_events(world, tick)``.
    """
    cfg = world.config.semantic
    result = SemanticPassResult()
    if not cfg.enabled:
        return result

    should, trigger = should_run_semantic_pass(world, tick)
    if not should and not force:
        return result

    lens_keys = lenses_with_pending_buffers(world)
    if not lens_keys:
        return result

    adapter = adapter or get_semantic_adapter(world, lm_adapter=lm_adapter)
    combined = SemanticCompileResult()

    for key in lens_keys:
        slice_ = build_slice(world, key, trigger=trigger)
        if not slice_.canonical_events:
            clear_lens_buffer(world, key)
            continue
        deltas = adapter.infer_pressure(slice_)
        if not deltas:
            clear_lens_buffer(world, key)
            continue
        part = compile_deltas(world, slice_, deltas)
        combined.transitions.extend(part.transitions)
        combined.log_lines.extend(part.log_lines)
        combined.applied_count += part.applied_count
        result.lenses_processed += 1
        clear_lens_buffer(world, key)

    if combined.transitions:
        apply_semantic_transitions(world, combined.transitions, tick=tick)

    state = ensure_semantic_state(world)
    state["last_pass_tick"] = tick
    result.ran = True
    result.trigger = trigger.value
    result.deltas_applied = combined.applied_count
    result.log_lines = combined.log_lines
    return result


def on_tick_end(
    world: WorldState,
    tick: int,
    *,
    ingest_all_new: bool = False,
    lm_adapter: Optional["LMAdapter"] = None,
) -> SemanticPassResult:
    """Record events into buffers and maybe run semantic pass."""
    if ingest_all_new:
        ingest_new_events(world)
    else:
        record_tick_events(world, tick)
    return run_semantic_dynamics_pass(world, tick, lm_adapter=lm_adapter)
