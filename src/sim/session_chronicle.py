"""
Story-mode presentation for autonomous simulation.

Formats tick headers, scenario chapter breaks, filtered action lines, and
end-of-session recaps — without mutating canonical world state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from .game_loop import StepResult
    from .schemas import WorldState


def _scenario(world: "WorldState") -> dict:
    return (world.config.extra or {}).get("scenario") or {}


def current_beat(world: "WorldState") -> dict | None:
    beats: list[dict] = list(_scenario(world).get("beats") or [])
    if not beats:
        return None
    active: dict | None = None
    for beat in beats:
        if int(beat.get("tick", 0)) <= world.tick:
            active = beat
        else:
            break
    return active


def scene_tension(world: "WorldState") -> float:
    beat = current_beat(world)
    tension = min(1.0, len(world.event_log) / 40.0)
    if beat is not None and beat.get("tension") is not None:
        tension = max(tension, float(beat["tension"]))
    override = world.meta.get("director_tension_override")
    if override is not None:
        tension = float(override)
    return min(1.0, max(0.0, tension))


def _tension_bar(tension: float, width: int = 12) -> str:
    filled = int(round(tension * width))
    return "█" * filled + "░" * (width - filled)


def format_beat_chapter(scenario_lines: list[str], world: "WorldState") -> list[str]:
    """
    Turn raw scenario executor lines into a chapter break when a beat starts.
    """
    if not scenario_lines:
        return []

    beat = current_beat(world)
    if beat is None:
        return [f"  {line}" for line in scenario_lines]

    phase = beat.get("phase", world.meta.get("scenario_phase", 0))
    title = beat.get("title") or _scenario(world).get("title") or "Scene"
    note = beat.get("note") or beat.get("focus") or ""
    tension = scene_tension(world)

    lines = [
        "",
        f"{'─'*56}",
        f"  Chapter {phase}: {title}",
        f"  Tension [{_tension_bar(tension)}] {tension:.0%}",
    ]
    if note:
        lines.append(f"  {note}")
    lines.append(f"{'─'*56}")
    lines.append("")
    return lines


def format_tick_header(world: "WorldState", tick: int) -> str:
    beat = current_beat(world)
    phase = beat.get("phase") if beat else world.meta.get("scenario_phase", 0)
    tension = scene_tension(world)
    location = world.meta.get("location_name", world.name)
    return (
        f"── tick {tick} · {location} · phase {phase} "
        f"[{_tension_bar(tension, 8)}] ──"
    )


def _is_mundane_wait(result: "StepResult") -> bool:
    if result.event is None or result.action is None:
        return False
    verb = str(result.action.verb).lower()
    if verb != "wait":
        return False
    if result.event.transitions:
        return False
    return True


def format_action_line(
    result: "StepResult",
    world: "WorldState",
    *,
    story_mode: bool = False,
) -> Optional[str]:
    """Return a display line for one entity action, or None to suppress."""
    from .narrator import render_event

    if story_mode and _is_mundane_wait(result):
        return None

    if result.event is None:
        return f"  {result.status}"

    actor_id = result.action.actor if result.action else None
    actor_name = "Someone"
    if actor_id:
        ent = world.spatial.entities.get(actor_id)
        if ent:
            actor_name = ent.name

    narration = render_event(result.event, world)
    if story_mode:
        return f"  {actor_name}: {narration}"
    return f"  {result.status}\n    {narration}"


def build_session_recap(world: "WorldState") -> list[str]:
    """
    Summarise the simulation as story beats for the end banner.
    """
    from .schemas import TransitionKind

    deaths: list[str] = []
    goals: list[str] = []
    violence: list[str] = []
    dialogue: list[str] = []
    secrets: list[str] = []

    grid = world.spatial

    def _name(eid: str) -> str:
        ent = grid.entities.get(eid)  # type: ignore[arg-type]
        return ent.name if ent else eid

    for ev in world.event_log:
        verb = str(ev.action.verb).lower()
        actor = _name(str(ev.action.actor))

        for t in ev.transitions:
            kind = t.kind
            if kind == TransitionKind.ENTITY_DIED:
                deaths.append(f"{_name(t.payload['entity_id'])} fell")
            elif kind == TransitionKind.ENTITY_GOAL_ADDED:
                goal = t.payload.get("goal", "")
                if goal and goal not in goals:
                    goals.append(f"{_name(t.payload.get('entity_id', ''))}: {goal[:70]}")
            elif kind == TransitionKind.DIALOGUE_SPOKEN:
                text = str(t.payload.get("text", "")).strip()
                if text and len(dialogue) < 6:
                    dialogue.append(f'{actor}: "{text[:80]}"')
            elif kind == TransitionKind.ENTITY_HEALTH_CHANGED:
                if t.payload.get("cause") in ("combat", "attack"):
                    target = _name(str(ev.action.target or ""))
                    if target and f"{actor} vs {target}" not in violence:
                        violence.append(f"{actor} struck {target}")
            elif kind in (TransitionKind.EDGE_CREATED, TransitionKind.EDGE_UPDATED):
                ek = str(t.payload.get("edge_kind", ""))
                if "distrust" in ek or "enemy" in ek or "rival" in ek:
                    tgt = _name(str(t.payload.get("target", "")))
                    secrets.append(f"{actor} ↔ {tgt} ({ek.replace('_', ' ')})")

        if verb in ("attack", "fight", "threaten") and not violence:
            target = _name(str(ev.action.target or ""))
            if target:
                violence.append(f"{actor} {verb} {target}")

    lines = ["  Story recap:"]
    if dialogue:
        lines.append("  Dialogue highlights:")
        lines.extend(f"    · {d}" for d in dialogue[:4])
    if violence:
        lines.append("  Conflict:")
        lines.extend(f"    · {v}" for v in violence[:4])
    if deaths:
        lines.append("  Fallen:")
        lines.extend(f"    · {d}" for d in deaths)
    if goals:
        lines.append("  Goals in play:")
        lines.extend(f"    · {g}" for g in goals[:4])
    if secrets:
        lines.append("  Relationships shifted:")
        lines.extend(f"    · {s}" for s in secrets[:3])

    if world.world_facts:
        lines.append("  World remembers:")
        for fact in world.world_facts[-4:]:
            lines.append(f"    · {fact.claim[:90]}")

    if world.player_chronicle:
        lines.append("  Your story so far:")
        from .episodic_memory import chronicle_lines

        for entry in chronicle_lines(world, max_entries=4):
            lines.append(f"    · {entry[:100]}")

    beat = current_beat(world)
    if beat:
        lines.append(
            f"  Final scene: phase {beat.get('phase', '?')} — "
            f"{beat.get('note') or beat.get('focus') or 'unresolved'}"
        )

    if len(lines) == 1:
        lines.append("    (Quiet session — try a longer run or richer scenario beats.)")
    return lines


def format_scene_opening(world: "WorldState") -> list[str]:
    """
    Opening banner for interactive play — scene title, phase, and DM focus.
    """
    scenario = _scenario(world)
    beat = current_beat(world)
    title = scenario.get("title") or world.name
    if not beat and not scenario.get("title"):
        return []

    lines = [
        "",
        f"{'═'*56}",
        f"  {title}",
    ]
    if beat:
        phase = beat.get("phase", world.meta.get("scenario_phase", 0))
        tension = scene_tension(world)
        note = beat.get("focus") or beat.get("note") or ""
        lines.append(
            f"  Phase {phase} · Tension [{_tension_bar(tension)}] {tension:.0%}"
        )
        if note:
            lines.append(f"  {note}")
    lines.extend([f"{'═'*56}", ""])
    return lines


def format_rejection_hint(result: "StepResult") -> Optional[str]:
    """
    Player-friendly hint when an action fails or only partially resolves.
    """
    from .schemas import ActionZone, RejectionReason

    if result.narration and result.validation.valid:
        return None

    status = result.status or ""
    if "[ADJUDICATED]" in status and result.narration:
        return None

    if result.grounding.zone == ActionZone.ORPHANED:
        return (
            "Nothing in the world matched that — try naming someone you can see, "
            "a simpler physical action, or speak directly: say to Mira: …"
        )

    reason = result.validation.rejection_reason
    detail = (result.validation.rejection_detail or "").lower()

    hints: dict[RejectionReason, str] = {
        RejectionReason.TARGET_NOT_FOUND: (
            "That target isn't here — check the map or use observe to see who's nearby."
        ),
        RejectionReason.ENTITY_NOT_VISIBLE: (
            "You can't reach or see them from here — move closer or change facing."
        ),
        RejectionReason.PATH_INACCESSIBLE: (
            "The way is blocked — try another route or inspect what's in the way."
        ),
        RejectionReason.TARGET_OUT_OF_RANGE: (
            "Too far away — move closer first."
        ),
        RejectionReason.PHYSICALLY_IMPOSSIBLE: (
            "The engine can't do that as stated — try a smaller step or different verb."
        ),
        RejectionReason.MALFORMED_ACTION: (
            "Couldn't parse that intent — try shorter phrasing or a direct command "
            "(move north, say to X: …)."
        ),
        RejectionReason.ACTION_TYPE_UNSUPPORTED: (
            "That action type isn't supported here — try rephrasing or a social approach."
        ),
    }
    if reason in hints:
        return hints[reason]

    if "wait" in detail or "malformed" in detail:
        return hints[RejectionReason.MALFORMED_ACTION]

    if not result.validation.valid and not result.narration:
        return "The world didn't accept that as-is — try rephrasing or approach an NPC directly."

    return None


def apply_interactive_turn(
    result: "StepResult",
    world: "WorldState",
    emit: Callable[[str, str], None],
    *,
    story_mode: bool = False,
) -> None:
    """
    Unified turn presentation for REPL, TUI, and pygame clients.
    """
    from .narrator import render_event
    from .schemas import ActionZone
    from .social_stimulus import format_player_feedback

    # Scenario chapter break
    if result.scenario_lines:
        if story_mode:
            for line in format_beat_chapter(result.scenario_lines, world):
                emit(line, "ambient")
        else:
            for line in result.scenario_lines:
                emit(f"  {line}", "ambient")

    if story_mode:
        emit(format_tick_header(world, result.tick), "dim")
        emit("", "normal")

    # Primary narration (accepted, adjudicated, or zone 3)
    if result.narration:
        for i, line in enumerate(result.narration.splitlines()):
            col = "good" if i == 0 else "normal"
            emit(f"  {line}", col)
    elif result.event:
        narration = render_event(result.event, world)
        for i, line in enumerate(narration.splitlines()):
            emit(f"  {line}", "good" if i == 0 else "normal")

    # Technical status (hidden in story mode unless failure)
    show_status = not story_mode or (
        not result.validation.valid
        and "[ADJUDICATED]" not in (result.status or "")
    )
    if show_status and result.status:
        col = "dim" if result.validation.valid else "reject"
        emit(f"  {result.status}", col)

    hint = format_rejection_hint(result)
    if hint:
        emit(f"  → {hint}", "reject")

    if (
        result.grounding.zone == ActionZone.EXTENDING
        and result.validation.valid
        and result.grounding.asserted_fact
    ):
        emit(
            "  [The world registers this claim — whether it's believed "
            "depends on what just happened.]",
            "dim",
        )

    if result.stimulus_feedback:
        feedback = format_player_feedback(result.stimulus_feedback)
        if feedback:
            emit(feedback, "dim")

    for amb in result.ambient_events:
        hint_text = amb.narrative_hint or "(ambient event)"
        prefix = "[WORLD]" if not story_mode else "·"
        emit(f"  {prefix} {hint_text}", "ambient")

    for npc_res in result.npc_results:
        if story_mode and _is_mundane_wait(npc_res):
            continue
        if npc_res.narration:
            for line in npc_res.narration.splitlines():
                emit(f"    {line}", "npc")
        elif npc_res.event:
            npc_line = render_event(npc_res.event, world)
            for line in npc_line.splitlines():
                col = "npc" if ":" in line else "dim"
                emit(f"    {line}", col)
        elif not story_mode:
            emit(f"  {npc_res.status}", "dim")

    for g in result.completed_goals:
        emit(f"  ★ Quest complete: {g.title}!", "good")
        if getattr(g, "description", None):
            emit(f"    {g.description}", "dim")
        reward = getattr(g, "reward", None)
        if reward and getattr(reward, "narrative", None):
            emit(f"    {reward.narrative}", "dim")

    for line in result.pressure_lines or []:
        emit(f"  {line}", "dim")

    for line in result.campaign_lines or []:
        emit(f"  {line}", "ambient")

    if result.quest_journal:
        from .quest_journal import format_journal_lines

        journal_lines = format_journal_lines(result.quest_journal)
        if journal_lines:
            if story_mode:
                emit("  — Journal —", "dim")
            for line in journal_lines[:4]:
                emit(f"  📜 {line}" if story_mode else f"  {line}", "dim")
