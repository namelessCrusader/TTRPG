"""
Deterministic social follow-ups after CONTACT actions.

Complements compiler disposition changes with relational edge nudges and
scenario phase flags so intimacy/combat contact reshapes the scene.
"""

from __future__ import annotations

from .schemas import (
    ConsentState,
    EdgeKind,
    Transition,
    TransitionKind,
    WorldState,
    Event,
)
from .compiler import apply_transitions

_ROMANTIC_VERBS = frozenset({
    "kiss", "embrace", "hug", "caress", "cuddle", "hold", "nuzzle",
})
_AGGRESSIVE_VERBS = frozenset({
    "punch", "slap", "shove", "grab", "push",
})


def apply_contact_social_effects(world: WorldState, event: Event) -> list[str]:
    """Apply edge/meta updates for CONTACT_INITIATED; return log lines."""
    lines: list[str] = []
    verb = str(event.action.verb).lower().split(".")[-1]
    for t in event.transitions:
        if t.kind != TransitionKind.CONTACT_INITIATED:
            continue
        actor = str(t.payload.get("actor", ""))
        target = str(t.payload.get("target", ""))
        consent = str(t.payload.get("consent_state", ""))
        if not actor or not target:
            continue

        transitions: list[Transition] = []
        if consent == ConsentState.WELCOMED.value and verb in _ROMANTIC_VERBS:
            transitions.extend(
                _intimate_nudge(actor, target, delta=0.15, tick=world.tick)
            )
            target_ent = world.spatial.entities.get(target) or world.all_entities().get(
                target
            )
            if target_ent is not None:
                target_ent.meta["beat_first_contact"] = world.tick
            lines.append(f"[contact] warmth between companions (+intimate)")

        elif consent in (
            ConsentState.UNWELCOMED.value,
            ConsentState.REFUSED.value,
        ):
            transitions.extend(
                _edge_nudge_transitions(
                    actor, target, EdgeKind.WARY_OF.value, 0.55, world.tick
                )
            )
            lines.append(f"[contact] boundary pushed (+wary)")

        elif consent == ConsentState.HOSTILE.value or verb in _AGGRESSIVE_VERBS:
            transitions.extend(
                _edge_nudge_transitions(
                    actor, target, EdgeKind.THREATENED.value, 0.6, world.tick
                )
            )
            lines.append(f"[contact] hostility registered (+threatened)")

        if transitions:
            apply_transitions(world, transitions)

        # Advance scenario phase on meaningful contact
        if verb in _ROMANTIC_VERBS and consent == ConsentState.WELCOMED.value:
            phase = int(world.meta.get("scenario_phase", 0))
            if phase < 2:
                world.meta["scenario_phase"] = 2
        elif consent == ConsentState.HOSTILE.value:
            world.meta["scenario_phase"] = max(
                int(world.meta.get("scenario_phase", 0)), 3
            )

    return lines


def _intimate_nudge(
    source: str, target: str, *, delta: float, tick: int
) -> list[Transition]:
    return _edge_nudge_transitions(
        source, target, EdgeKind.INTIMATE.value, 0.35 + delta, tick
    ) + _edge_nudge_transitions(
        target, source, EdgeKind.INTIMATE.value, 0.35 + delta, tick
    )


def _edge_nudge_transitions(
    source: str, target: str, edge_kind: str, weight: float, tick: int
) -> list[Transition]:
    return [
        Transition(
            kind=TransitionKind.EDGE_CREATED,
            payload={
                "from_entity": source,
                "to_entity": target,
                "edge_kind": edge_kind,
                "weight": weight,
                "since_tick": tick,
            },
        )
    ]
