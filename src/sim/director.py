"""
Narrative Director — deterministic scene guidance for NPC cognition.

Does not mutate canonical world state. Produces hints consumed by
NpcCharacterSheet and LM prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .speech_utils import collect_recent_room_dialogue, normalize_speech_key

if TYPE_CHECKING:
    from .schemas import EntityId, EntityState, WorldState


@dataclass
class DirectorBrief:
    """Per-tick guidance for one NPC."""

    scene_tension: float = 0.0
    focus_hint: str = ""
    objective: str = ""
    taboo_phrases: list[str] = field(default_factory=list)
    allow_infer: bool = False
    prefer_non_speak: bool = False


class NarrativeDirector:
    """
    Reads event log + goals and emits policy/LM constraints.

    Scene beats can be loaded from ``world.config.extra["scenario"]`` or
    ``worlds/<pack>/scenarios/<name>.yaml``.
    """

    def __init__(self, world: "WorldState") -> None:
        self._world = world
        self._scenario = (world.config.extra or {}).get("scenario") or {}
        self._beats: list[dict] = list(self._scenario.get("beats") or [])

    def _current_beat(self) -> Optional[dict]:
        tick = self._world.tick
        active: dict | None = None
        for beat in self._beats:
            if int(beat.get("tick", 0)) <= tick:
                active = beat
            else:
                break
        return active

    def brief_for_npc(self, entity: "EntityState") -> DirectorBrief:
        w = self._world
        eid = str(entity.entity_id)
        tension = min(1.0, len(w.event_log) / 40.0)
        beat = self._current_beat()
        if beat is not None:
            if beat.get("tension") is not None:
                tension = max(tension, float(beat["tension"]))
            w.meta["scenario_phase"] = int(beat.get("phase", w.meta.get("scenario_phase", 0)))

        taboo: list[str] = []
        for line in collect_recent_room_dialogue(w, max_lines=5):
            if ": \"" in line:
                frag = line.split(": \"", 1)[-1].rstrip("\"")
                if frag:
                    taboo.append(frag[:60])

        # Over-speech detection: if this NPC spoke last 2 turns, push variety
        recent_self_speak = 0
        for ev in reversed(w.event_log[-6:]):
            if str(ev.action.actor) == eid:
                v = str(ev.action.verb).lower()
                if v in ("speak", "ask", "say", "sing"):
                    recent_self_speak += 1

        infer_budget = int(entity.meta.get("director_infer_budget", 0))
        allow_infer = infer_budget > 0

        drive_snip = (entity.drive or "").split(";")[0].strip()[:50]
        focus = f"Advance: {drive_snip}" if drive_snip else "Stay in character."
        if beat and beat.get("focus"):
            focus = str(beat["focus"])
        elif beat and beat.get("note"):
            focus = str(beat["note"])
        scene_focus = w.meta.get("director_scene_focus")
        if scene_focus:
            focus = str(scene_focus)
        if w.meta.get("director_tension_override") is not None:
            tension = float(w.meta["director_tension_override"])

        for pres in w.meta.get("active_pressures") or []:
            if pres.get("director_focus"):
                focus = str(pres["director_focus"])

        objective = self._objective_for(entity, focus, tension)

        return DirectorBrief(
            scene_tension=tension,
            focus_hint=focus,
            objective=objective,
            taboo_phrases=taboo[:5],
            allow_infer=allow_infer,
            prefer_non_speak=recent_self_speak >= 2,
        )

    def _objective_for(
        self,
        entity: "EntityState",
        focus: str,
        tension: float,
    ) -> str:
        """Concrete DM beat for this character's next turn."""
        w = self._world
        others = [
            e for e in w.spatial.entities.values()
            if e.alive and e.entity_id != entity.entity_id
        ]
        others.sort(key=lambda e: entity.position.manhattan(e.position))
        target = others[0] if others else None
        target_name = target.name if target else "someone nearby"

        lower_focus = focus.lower()
        role = (entity.role or entity.occupation or "").lower()
        name = entity.name.lower()
        has_secret = bool(entity.secrets)

        if "secrets" in lower_focus or "curiosity" in lower_focus:
            if has_secret:
                return (
                    f"Move the secret economy forward: test whether {target_name} "
                    "can be trusted, extract leverage, or reveal a guarded half-truth."
                )
            if "bard" in role or "linna" in name:
                return (
                    f"Pull information from {target_name}: eavesdrop, perform as cover, "
                    "or ask a pointed question that exposes a secret."
                )
            return (
                f"Pressure {target_name} toward a specific secret, debt, or contradiction. "
                "Do not make small talk."
            )

        if "alliance" in lower_focus or "distrust" in lower_focus or tension >= 0.65:
            return (
                f"Force a social choice involving {target_name}: accuse, defend, bargain, "
                "or demand a concrete concession."
            )

        if "settle" in lower_focus or "mood" in lower_focus or tension < 0.35:
            if "bartender" in role or "mira" in name:
                return (
                    f"Establish control of the room through {target_name}: pour a drink "
                    "only if a receptacle is valid, greet, warn, or redirect tension."
                )
            if "bard" in role or "linna" in name:
                return (
                    f"Set the room's tone around {target_name}: perform, tease, or "
                    "listen for the first crack in the mood."
                )
            return (
                f"Create an opening beat with {target_name}: greet, haggle, probe, "
                "or move closer to a concrete social aim."
            )

        drive = (entity.drive or "").split(";")[0].strip()
        if drive:
            return f"Advance this drive in a visible way with {target_name}: {drive[:120]}"
        return f"Make a consequential choice involving {target_name}; change a relationship or fact."

    def tick_start(self, world: "WorldState") -> list[str]:
        """
        Refresh per-NPC infer budgets and apply one-shot scenario beat effects.

        Returns log lines from scenario entry (empty if beat unchanged).
        """
        from .scenario_executor import apply_scenario_entry_effects

        scenario_lines = apply_scenario_entry_effects(world)

        cfg = world.config.npc_policy.cognition
        max_infer = cfg.max_infer_per_npc
        for ent in world.all_entities().values():
            if not ent.alive:
                continue
            budget = int(ent.meta.get("director_infer_budget", 0))
            # Regenerate budget slowly
            if world.tick % max(1, cfg.infer_cooldown_ticks) == 0:
                ent.meta["director_infer_budget"] = max_infer
            elif budget < max_infer:
                ent.meta["director_infer_budget"] = min(
                    max_infer, budget + 1
                )
        return scenario_lines

    def consume_infer_budget(self, entity: "EntityState") -> None:
        b = int(entity.meta.get("director_infer_budget", 0))
        if b > 0:
            entity.meta["director_infer_budget"] = b - 1

    def format_hint_block(self, brief: DirectorBrief) -> str:
        lines = []
        if brief.focus_hint:
            lines.append(f"Scene focus: {brief.focus_hint}")
        if brief.objective:
            lines.append(f"DM objective this turn: {brief.objective}")
        if brief.taboo_phrases:
            lines.append(
                "Do NOT repeat: "
                + "; ".join(f'"{t}"' for t in brief.taboo_phrases[:3])
            )
        if brief.prefer_non_speak:
            lines.append(
                "You spoke recently — prefer move, observe, or a non-verbal action."
            )
        return "\n".join(lines)
