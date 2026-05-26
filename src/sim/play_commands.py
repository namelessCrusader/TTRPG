"""
Shared meta-commands and turn result formatting for all front-ends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Protocol, Union

from .voxel_layers import LayerNavigator, layer_help_lines, render_layer_ascii


class MessageSink(Protocol):
    def __call__(self, text: str, color: str = "normal") -> None: ...


SessionLike = Union[object, None]


# Terminal REPL uses plain print; pygame clients map colors to RGB tuples.
COLOR_MAP = {
    "normal": None,
    "dim": "dim",
    "ambient": "ambient",
    "player": "player",
    "npc": "npc",
    "reject": "reject",
    "good": "good",
}


def _maybe_autosave(session: SessionLike) -> None:
    if session is None:
        return
    ckpt = getattr(session, "checkpoint_dir", None)
    every = int(getattr(session, "autosave_every", 0) or 0)
    if not ckpt or every <= 0:
        return
    world = session.world
    if world.tick <= 0 or world.tick % every != 0:
        return
    from .lm_budget import metrics_snapshot
    from .persistence import save_checkpoint

    save_checkpoint(
        world,
        ckpt,
        pack_id=world.meta.get("pack_id"),
        lm_metrics=metrics_snapshot(world),
    )


def handle_persistence_command(
    raw: str,
    *,
    session: SessionLike,
    world,
    loop,
    emit,
) -> bool:
    """
    Unified save / load / checkpoint / chronicle / metrics / stream commands.

    Returns True if handled.
    """
    lower = raw.lower().strip()
    parts = raw.split(None, 2)

    if lower.startswith("save"):
        filename = parts[1].strip() if len(parts) > 1 else "savegame.json"
        try:
            from .lm_budget import metrics_snapshot
            from .persistence import save_world

            n = save_world(
                world,
                filename,
                pack_id=world.meta.get("pack_id"),
                lm_metrics=metrics_snapshot(world),
            )
            emit(f"[SAVED] {filename} (tick={world.tick}, {n} events)", "good")
        except Exception as exc:
            emit(f"Could not save: {exc}", "reject")
        return True

    if lower.startswith("load "):
        filename = raw.split(None, 1)[1].strip()
        try:
            from .cli import PlaySession, recreate_loop
            from .persistence import load_world

            pack = None
            rebind = False
            if isinstance(session, PlaySession):
                pack = str(session.world_path)
                rebind = bool(world.meta.get("pack_rebound_at") is not None)
            loaded = load_world(
                filename,
                pack_path=pack,
                rebind_config=rebind,
            )
            if session is not None and hasattr(session, "loop"):
                recreate_loop(session, loaded)
                loop = session.loop
                world = session.world
            emit(
                f"[LOADED] {filename} (tick={world.tick}, "
                f"{len(world.event_log)} events)",
                "good",
            )
        except Exception as exc:
            emit(f"Could not load: {exc}", "reject")
        return True

    if lower == "checkpoint" or lower.startswith("checkpoint "):
        if session is None or not getattr(session, "checkpoint_dir", None):
            emit("No checkpoint directory configured (--checkpoint-dir).", "dim")
            return True
        try:
            from .lm_budget import metrics_snapshot
            from .persistence import save_checkpoint

            path = save_checkpoint(
                world,
                session.checkpoint_dir,
                pack_id=world.meta.get("pack_id"),
                lm_metrics=metrics_snapshot(world),
            )
            emit(f"[CHECKPOINT] {path.name} tick={world.tick}", "good")
        except Exception as exc:
            emit(f"Checkpoint failed: {exc}", "reject")
        return True

    if lower == "chronicle":
        from .episodic_memory import chronicle_lines

        lines = chronicle_lines(world, max_entries=8)
        if not lines:
            emit("(no chronicle entries yet — play ~20 ticks)", "dim")
        for line in lines:
            emit(f"  · {line}", "normal")
        return True

    if lower == "metrics" or lower == "lm":
        from .lm_budget import format_metrics_summary

        emit(format_metrics_summary(world), "normal")
        return True

    if lower == "stream" or lower.startswith("stream "):
        collector = getattr(loop, "presentation", None)
        if collector is None:
            emit("Presentation stream not enabled.", "dim")
            return True
        out = parts[1].strip() if len(parts) > 1 else "frames.json"
        try:
            Path(out).write_text(collector.to_json(), encoding="utf-8")
            emit(
                f"[STREAM] Wrote {len(collector.stream.frames)} frames → {out}",
                "good",
            )
        except Exception as exc:
            emit(f"Stream export failed: {exc}", "reject")
        return True

    return False


def handle_meta_command(
    raw: str,
    *,
    world,
    loop,
    player_id: Optional[str],
    layers: Optional[LayerNavigator],
    emit,
    pygame_quit=None,
) -> bool:
    """
    Handle commands that do not advance the simulation.

    ``emit(text, color)`` — color is a logical name (see COLOR_MAP).
    Returns True if handled.
    """
    lower = raw.lower().strip()

    if lower in ("quit", "exit", "q"):
        if pygame_quit:
            import pygame
            pygame.event.post(pygame.event.Event(pygame.QUIT))
        return True

    if layers is not None:
        layer_msg = layers.handle_repl_command(raw)
        if layer_msg is not None:
            emit(f"Layer: {layer_msg}", "ambient")
            if layers.is_slice_mode and layers.slice_z is not None:
                for line in render_layer_ascii(
                    world.spatial, layers.slice_z, player_id=player_id
                ).splitlines():
                    emit(line, "dim")
            return True

    if lower in ("inventory", "inv", "i"):
        from .narrator import render_inventory
        player = world.spatial.entities.get(player_id) if player_id else None
        if player:
            for line in render_inventory(player, world).splitlines():
                emit(line, "normal")
        else:
            emit("(no player entity)", "dim")
        return True

    if lower.startswith("inspect") or lower.startswith("examine"):
        from .player_inspect import format_inspect_report

        target = None
        for prefix in ("inspect ", "examine ", "look at "):
            if lower.startswith(prefix):
                target = raw[len(prefix):].strip() or None
                break
        report = format_inspect_report(world, player_id, target)
        for line in report.splitlines():
            emit(line, "normal")
        return True

    if lower.startswith("social") or lower == "rumours" or lower == "rumors":
        from .social_inspect import format_social_inspect

        target = raw.split(None, 1)[1].strip() if lower.startswith("social ") and len(raw.split()) > 1 else None
        report = format_social_inspect(world, player_id, target)
        for line in report.splitlines():
            emit(line, "normal")
        return True

    if lower == "journal" or lower == "quests":
        from .quest_journal import build_quest_journal, format_journal_lines

        entries = build_quest_journal(world, player_id)
        lines = format_journal_lines(entries)
        if not lines:
            emit("(journal empty — play a while)", "dim")
        for line in lines:
            emit(line, "normal")
        return True

    if lower in ("map", "m"):
        grid = world.spatial
        if grid.use_voxels or grid.depth > 1:
            z = 0
            if layers and layers.is_slice_mode and layers.slice_z is not None:
                z = layers.slice_z
            elif player_id and player_id in grid.entities:
                z = grid.entities[player_id].position.z
            for line in render_layer_ascii(grid, z, player_id=player_id).splitlines():
                emit(line, "dim" if grid.use_voxels else "normal")
        else:
            from .narrator import render_map
            emit(render_map(world, player_id), "normal")
        return True

    if lower == "status":
        from .narrator import render_world_summary
        for line in render_world_summary(world).splitlines():
            emit(line, "normal")
        return True

    if lower == "log":
        from .narrator import render_event_log
        lines = render_event_log(world)
        for line in (lines[-20:] if lines else ["(empty)"]):
            emit(line, "dim")
        return True

    if lower == "goals":
        goals = getattr(world, "goals", []) or getattr(world.config, "goals", [])
        if not goals:
            emit("No goals defined.", "dim")
        for g in goals:
            mark = "[done]" if getattr(g, "completed", False) else "[pending]"
            emit(f"  {mark} {g.title}: {g.description}", "normal")
        return True

    if lower == "help":
        for line in help_lines(layers is not None):
            emit(line, "dim")
        return True

    if handle_persistence_command(
        raw, session=None, world=world, loop=loop, emit=emit,
    ):
        return True

    return False


def help_lines(volumetric: bool) -> list[str]:
    lines = [
        "── Commands ──",
        "  inventory / inspect [target] / social [target] / journal / map / status / log / goals / chronicle / metrics",
        "  save [file] / load <file> / checkpoint / stream [file]",
        "  quit",
        "── Actions ──",
        "  Type natural language, e.g. 'say hello to the guard'",
    ]
    if volumetric:
        lines.extend(layer_help_lines())
    return lines


def apply_step_result(
    result,
    world,
    emit: Callable,
    *,
    story_mode: bool = False,
) -> None:
    """Format a GameLoop StepResult for any UI."""
    from .session_chronicle import apply_interactive_turn

    apply_interactive_turn(result, world, emit, story_mode=story_mode)


def run_player_turn(
    raw: str,
    loop,
    player_id,
    emit,
    on_world_changed=None,
    session: SessionLike = None,
    *,
    story_mode: bool = False,
    show_world_diff: bool = True,
) -> None:
    """Execute one game turn with optional debug diff and autosave."""
    snap = None
    world_snap = None
    if getattr(loop, "debug_mode", False):
        from .sim_debug import capture_snapshot
        snap = capture_snapshot(loop.world)
    if show_world_diff:
        from .world_diff import capture_player_snapshot
        world_snap = capture_player_snapshot(loop.world, player_id)

    try:
        result = loop.step(raw)
    except Exception as exc:
        emit(f"Error: {exc}", "reject")
        return

    apply_step_result(result, loop.world, emit, story_mode=story_mode)

    if world_snap is not None and show_world_diff:
        from .world_diff import capture_player_snapshot, format_world_diff

        transitions: list = []
        if result.validation and result.validation.concrete_transitions:
            transitions.extend(result.validation.concrete_transitions)
        if result.event and result.event.transitions:
            transitions.extend(result.event.transitions)
        diff = format_world_diff(
            world_snap,
            capture_player_snapshot(loop.world, player_id),
            transitions,
            loop.world,
        )
        if diff.strip():
            emit(diff, "good")

    if snap is not None and getattr(loop, "debug_mode", False):
        from .sim_debug import capture_snapshot, format_snapshot_diff
        diff = format_snapshot_diff(snap, capture_snapshot(loop.world))
        if diff.strip():
            emit(diff, "dim")

    _maybe_autosave(session)

    if on_world_changed:
        on_world_changed()
