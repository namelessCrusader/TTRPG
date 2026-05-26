"""
Two-layer narration: deterministic rules + optional LM color.

Rule layer always runs; LM layer only fills slots from canonical fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .narrator import render_event
from .schemas import Event, WorldState

if TYPE_CHECKING:
    from .lm_adapter import LMAdapter


def render_event_layered(
    event: Event,
    world: WorldState,
    *,
    adapter: Optional["LMAdapter"] = None,
    use_lm_color: bool = False,
) -> str:
    """
    Render narration for an event.

    Default: rule-based only (``narrator.render_event``).
    When ``use_lm_color`` and adapter provided, append a short LM flourish
    constrained to facts already in the event (experimental).
    """
    base = render_event(event, world)
    if not use_lm_color or adapter is None:
        return base
    try:
        from .lm_adapter import MockLMAdapter

        if isinstance(adapter, MockLMAdapter):
            return base
        prompt = (
            "Rewrite this simulation line in one vivid sentence. "
            "Do not add new facts or characters.\n"
            f"Line: {base}"
        )
        colored = adapter.narrate(prompt, world, event.action.actor)
        if colored and len(colored) < 300:
            return colored.strip()
    except Exception:
        pass
    return base
