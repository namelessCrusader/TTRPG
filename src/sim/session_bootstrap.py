"""
Shared session bootstrap helpers for REPL, TUI, and pygame clients.

All front-ends use the same rules for NPC policy:
  - MockLMAdapter  → ReactivePolicy (deterministic tests / no LM cost)
  - Real LM backend → LMNpcPolicy with ReactivePolicy fallback
  - --reactive-npcs → force ReactivePolicy even with a real adapter
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .lm_adapter import LMAdapter
    from .npc_policy import NpcPolicy
    from .schemas import WorldState


def resolve_npc_policy(
    adapter: "LMAdapter",
    world: "WorldState",
    *,
    force_reactive: bool = False,
) -> "NpcPolicy":
    """
    Pick the NPC policy all interactive clients should use.

    Matches ``GameLoop`` defaults so TUI/REPL/pygame behave identically.
    """
    from .lm_adapter import MockLMAdapter
    from .npc_lm_policy import LMNpcPolicy
    from .npc_policy import ReactivePolicy

    cfg = world.config.npc_policy
    reactive = ReactivePolicy(
        threat_lookback_ticks=cfg.threat_lookback_ticks,
        flee_health_fraction=cfg.flee_health_fraction,
    )
    if force_reactive or isinstance(adapter, MockLMAdapter):
        return reactive
    return LMNpcPolicy(adapter, fallback=reactive)


def build_game_loop(
    world: "WorldState",
    adapter: "LMAdapter",
    *,
    force_reactive_npcs: bool = False,
    debug_mode: bool = False,
    npc_policy: Optional["NpcPolicy"] = None,
):
    """Construct a ``GameLoop`` with unified NPC policy wiring."""
    from .game_loop import GameLoop

    policy = npc_policy or resolve_npc_policy(
        adapter, world, force_reactive=force_reactive_npcs,
    )
    return GameLoop(
        world,
        adapter,
        npc_policy=policy,
        npcs_act_each_turn=True,
        debug_mode=debug_mode,
    )


__all__ = ["build_game_loop", "resolve_npc_policy"]
