"""
Rich TUI frontend built with Textual.

Layout
------
┌────────────────────────────────────┬─────────────────┐
│                                    │                 │
│   NARRATIVE PANEL (scrollable)     │   MAP PANEL     │
│                                    │                 │
│                                    ├─────────────────┤
│                                    │  RELATIONS      │
│                                    │                 │
├────────────────────────────────────┴─────────────────┤
│  Status: tick=7  HP=100  mood=neutral  [day]         │
├──────────────────────────────────────────────────────┤
│ > What do you do?                                    │
└──────────────────────────────────────────────────────┘

Run via:
    python -m src.sim.tui [same flags as repl.py]
or:
    run_sim.sh --tui [world]
"""

from __future__ import annotations

import argparse
import json
from typing import Optional

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------


_CSS = """
Screen {
    layout: vertical;
}

#main-area {
    height: 1fr;
    layout: horizontal;
}

#narrative {
    width: 3fr;
    height: 100%;
    border: solid $primary-darken-2;
    padding: 0 1;
}

#right-side {
    width: 2fr;
    min-width: 42;
    height: 100%;
    layout: vertical;
}

#map-panel {
    height: 3fr;
    border: solid $accent-darken-2;
    padding: 0 1;
    overflow: auto auto;
}

#relations-panel {
    height: 2fr;
    border: solid $warning-darken-2;
    padding: 0 1;
    overflow: auto auto;
}

#status-bar {
    height: 3;
    background: $surface;
    color: $text-muted;
    padding: 0 2;
    border-top: solid $primary-darken-2;
    content-align: left middle;
}

#cmd-input {
    dock: bottom;
    height: 3;
}
"""


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class SimApp(App[None]):
    """Textual application wrapping the SimulationGameLoop."""

    CSS = _CSS
    TITLE = "Semantic Simulation Runtime"
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+m", "refresh_map", "Refresh map"),
        ("pageup", "layer_up", "Layer up"),
        ("pagedown", "layer_down", "Layer down"),
    ]

    def __init__(self, loop, player_id, adapter, world, *, story_mode: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.loop = loop
        self.player_id = player_id
        self.adapter = adapter
        self.world = world
        self.story_mode = story_mode
        self._busy = False
        from .voxel_layers import LayerNavigator
        g = world.spatial
        self._layer_nav = LayerNavigator(
            depth=max(1, g.depth),
            slice_z=0 if (g.use_voxels or g.depth > 1) else None,
        )
        self._voxel_map = bool(g.use_voxels or g.depth > 1)

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-area"):
            yield RichLog(id="narrative", highlight=True, markup=True, wrap=True)
            with Vertical(id="right-side"):
                yield Static("", id="map-panel")
                yield Static("", id="relations-panel")
        yield Static("", id="status-bar")
        yield Input(
            placeholder="What do you do? (type 'help' for commands)",
            id="cmd-input",
        )
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_map()
        self._refresh_status()
        self._refresh_relations()
        self.query_one("#cmd-input", Input).focus()
        # Welcome message
        extra = ""
        if self._voxel_map:
            extra = (
                f"Volume: {self.world.spatial.width}×"
                f"{self.world.spatial.height}×{self.world.spatial.depth}  "
                f"{self._layer_nav.label()}\n"
                "[dim]PgUp/PgDn or layer up/down — Z slices[/dim]\n"
            )
        self._log_narrative(
            f"[bold green]=== {self.world.name} ===[/bold green]\n"
            f"Adapter: [cyan]{_adapter_label(self.adapter)}[/cyan]\n"
            f"{extra}"
            f"Type [bold]help[/bold] for commands.\n"
        )
        from .session_chronicle import format_scene_opening

        for line in format_scene_opening(self.world):
            self._log_narrative(line)

    # ------------------------------------------------------------------
    # Input handler
    # ------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        raw = event.value.strip()
        event.input.clear()
        if not raw:
            return

        lower = raw.lower()

        if lower in ("quit", "exit", "q"):
            self.exit()
            return

        if lower == "help":
            self._log_narrative(_HELP_TEXT)
            return

        if lower in ("map", "m"):
            self._refresh_map()
            self._log_narrative("[dim]Map refreshed.[/dim]")
            return

        layer_msg = self._layer_nav.handle_repl_command(raw)
        if layer_msg is not None:
            self._refresh_map()
            self._refresh_status()
            self._log_narrative(f"[dim]Layer: {layer_msg}[/dim]")
            if self._layer_nav.is_slice_mode and self._layer_nav.slice_z is not None:
                from .voxel_layers import render_layer_ascii
                self._log_narrative(
                    render_layer_ascii(
                        self.world.spatial,
                        self._layer_nav.slice_z,
                        player_id=self.player_id,
                    )
                )
            return

        if lower == "status":
            from .narrator import render_world_summary
            self._log_narrative(render_world_summary(self.world))
            return

        if lower == "projection":
            from .projection import project
            proj = project(self.world, self.player_id)
            text = json.dumps(json.loads(proj.model_dump_json()), indent=2)
            self._log_narrative(f"[dim]{text}[/dim]")
            return

        if lower == "log":
            from .narrator import render_event_log
            lines = render_event_log(self.world)
            self._log_narrative("\n".join(lines) if lines else "(no events yet)")
            return

        if lower == "verbs":
            self._log_narrative(_format_verbs(self.world))
            return

        if lower == "goals":
            self._log_narrative(_format_goals(self.world))
            return

        if lower.startswith("save") or lower.startswith("load "):
            from .play_commands import handle_persistence_command

            def _emit(msg: str, color: str = "normal") -> None:
                self._log_narrative(msg)

            if handle_persistence_command(
                raw, session=None, world=self.world, loop=self.loop, emit=_emit,
            ):
                if lower.startswith("load"):
                    self._refresh_map()
                    self._refresh_status()
                    self._refresh_relations()
            return

        # Normal game action — run in a worker thread so the UI stays responsive
        if self._busy:
            self._log_narrative("[yellow]Still processing last action…[/yellow]")
            return

        self._process_action(raw)

    # ------------------------------------------------------------------
    # Worker — runs GameLoop.step() in a background thread
    # ------------------------------------------------------------------

    @work(thread=True, exclusive=True)
    def _process_action(self, intent: str) -> None:
        """Execute one game turn in a background thread."""
        self.call_from_thread(self._set_busy, True)
        try:
            result = self.loop.step(intent)
        except Exception as exc:
            self.call_from_thread(
                self._log_narrative,
                f"[red]Error:[/red] {exc}",
            )
            return
        finally:
            self.call_from_thread(self._set_busy, False)

        self.call_from_thread(self._render_result, result)

    # ------------------------------------------------------------------
    # Render helpers (must be called from main thread via call_from_thread)
    # ------------------------------------------------------------------

    def _render_result(self, result) -> None:
        from .play_commands import apply_step_result

        _RICH = {
            "normal": "",
            "dim": "dim",
            "ambient": "yellow",
            "player": "cyan",
            "npc": "orange3",
            "reject": "red",
            "good": "green",
        }

        def _emit(text: str, color: str = "normal") -> None:
            if not text:
                self._log_narrative("")
                return
            tag = _RICH.get(color, "")
            if tag:
                self._log_narrative(f"[{tag}]{text}[/{tag}]")
            else:
                self._log_narrative(text)

        apply_step_result(result, self.world, _emit, story_mode=self.story_mode)
        self._log_narrative("")

        # Refresh panels
        self._refresh_map()
        self._refresh_status()
        self._refresh_relations()

    def _log_narrative(self, text: str) -> None:
        log = self.query_one("#narrative", RichLog)
        log.write(text)

    def _refresh_map(self) -> None:
        from .narrator import render_map
        from .voxel_layers import render_layer_ascii
        from rich.text import Text

        try:
            if self._voxel_map:
                z = (
                    self._layer_nav.slice_z
                    if self._layer_nav.is_slice_mode
                    else (
                        self.world.spatial.entities[self.player_id].position.z
                        if self.player_id
                        and self.player_id in self.world.spatial.entities
                        else 0
                    )
                )
                map_str = render_layer_ascii(
                    self.world.spatial, z, player_id=self.player_id
                )
            else:
                map_str = render_map(self.world, self.player_id, legend=False)
        except Exception:
            map_str = "(map unavailable)"
        # no_wrap=True prevents Rich from reflowing the ASCII art to the
        # widget width.  The panel has overflow: auto so users can scroll.
        self.query_one("#map-panel", Static).update(Text(map_str, no_wrap=True))

    def _refresh_status(self) -> None:
        tick = self.world.tick
        player_ent = (
            self.world.spatial.entities.get(self.player_id)
            if self.player_id else None
        )
        if player_ent:
            hp = player_ent.health
            mood = player_ent.emotional_state.value
        else:
            hp, mood = "?", "?"

        time_str = ""
        if self.world.config.clock:
            tod = self.world.config.clock.time_of_day(tick)
            time_str = f"  [{tod.value}]"

        conversations = [
            t for t in self.world.active_conversations
            if not t.closed
            and self.player_id
            and self.player_id in t.participant_ids
        ]
        convo_str = ""
        if conversations:
            partners = []
            for t in conversations[:2]:
                for pid, pname in zip(t.participant_ids, t.participant_names):
                    if pid != self.player_id:
                        partners.append(pname or pid)
            if partners:
                convo_str = f"  [talking: {', '.join(partners)}]"

        layer_str = ""
        if self._voxel_map:
            layer_str = f"  {self._layer_nav.label()}"

        self.query_one("#status-bar", Static).update(
            f"tick={tick}  HP={hp}  mood={mood}{time_str}{layer_str}{convo_str}"
        )

    def _refresh_relations(self) -> None:
        """
        Render the info panel: compact map legend + visible entities with
        their relational edges toward the player.
        """
        from .narrator import _LEGEND
        from .projection import project

        lines: list[str] = []

        # ── Compact legend ──────────────────────────────────────────────
        lines.append("[bold dim]LEGEND[/bold dim]")
        for sym, desc in _LEGEND:
            lines.append(f"  [bold]{sym}[/bold]  {desc}")
        lines.append("")

        # ── Visible entities + relationships ────────────────────────────
        if not self.player_id:
            lines.append("[dim](no player)[/dim]")
            self.query_one("#relations-panel", Static).update("\n".join(lines))
            return

        try:
            proj = project(self.world, self.player_id)
            if proj.visible_entities:
                lines.append("[bold dim]NEARBY[/bold dim]")
                for ent in proj.visible_entities:
                    # Edges from player → entity and entity → player
                    edges_out = (
                        self.world.relational.edges
                        .get(self.player_id, {})
                        .get(ent.entity_id, [])
                    )
                    edges_in = (
                        self.world.relational.edges
                        .get(ent.entity_id, {})
                        .get(self.player_id, [])
                    )
                    kinds: set[str] = set()
                    for e in list(edges_out) + list(edges_in):
                        kinds.add(
                            e.kind.value if hasattr(e.kind, "value") else str(e.kind)
                        )
                    rel_str = ", ".join(sorted(kinds)) if kinds else "stranger"
                    mood = ent.emotional_state.value
                    dist = ent.distance
                    alert = ent.alertness.value if hasattr(ent.alertness, "value") else str(ent.alertness)
                    lines.append(
                        f"[bold]{ent.name}[/bold]  {dist}t  "
                        f"[dim]{mood}/{alert}[/dim]"
                    )
                    lines.append(f"  [dim]{rel_str}[/dim]")
            else:
                lines.append("[dim](no entities in view)[/dim]")

            # Active goals summary
            goals = getattr(self.world.config, "goals", [])
            pending = [g for g in goals if not g.completed]
            if pending:
                lines.append("")
                lines.append("[bold dim]QUESTS[/bold dim]")
                for g in pending[:3]:
                    lines.append(f"  [yellow]◆[/yellow] {g.title}")

        except Exception as exc:
            lines.append(f"[red](error: {exc})[/red]")

        self.query_one("#relations-panel", Static).update("\n".join(lines))

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        widget = self.query_one("#cmd-input", Input)
        widget.placeholder = (
            "Processing…" if busy else "What do you do? (type 'help')"
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_refresh_map(self) -> None:
        self._refresh_map()

    def action_layer_up(self) -> None:
        if not self._voxel_map:
            return
        self._layer_nav.up()
        self._refresh_map()
        self._refresh_status()

    def action_layer_down(self) -> None:
        if not self._voxel_map:
            return
        self._layer_nav.down()
        self._refresh_map()
        self._refresh_status()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _adapter_label(adapter) -> str:
    name = type(adapter).__name__
    if "Mock" in name:
        return "Mock (no LM)"
    if "Ollama" in name:
        return "Ollama"
    if "OpenAI" in name:
        return "OpenAI"
    if "Outlines" in name:
        return "Outlines+llama.cpp"
    return name


def _format_verbs(world) -> str:
    kernel = sorted(["move", "attack", "take", "give", "throw"])
    lines = [
        "[bold]Kernel verbs[/bold] (bespoke physics):",
        *[f"  {v}" for v in kernel],
        "[bold]Pack-declared verb templates:[/bold]",
    ]
    for v in sorted(world.config.verb_templates.keys()):
        tmpl = world.config.verb_templates[v]
        contest = (
            f" (contest: {tmpl.contest.actor_attribute} vs "
            f"{tmpl.contest.target_attribute or 'difficulty'})"
            if tmpl.contest
            else ""
        )
        lines.append(f"  {v}{contest}")
    lines.append(
        "[dim]Any other verb is legal — engine uses built-in defaults "
        "or pure freeform path.[/dim]"
    )
    return "\n".join(lines)


def _format_goals(world) -> str:
    from .quest_journal import build_quest_journal, format_journal_lines

    player_id = next(
        (eid for eid, e in world.spatial.entities.items() if e.kind.value == "player"),
        None,
    )
    entries = build_quest_journal(world, player_id) if player_id else []
    if not entries:
        return "(no active objectives or concerns)"
    return "\n".join(format_journal_lines(entries))


def _check_newly_completed_goals(world) -> list:
    """Return goals that completed on this tick (for victory announcements)."""
    goals = getattr(world.config, "goals", [])
    return [g for g in goals if g.completed and g.completed_at == world.tick - 1]


_HELP_TEXT = """\
[bold]Commands:[/bold]
  map        — redraw the map (Z slice for voxel worlds)
  layer / layer N / layer up|down — Z height (voxel worlds)
  PgUp/PgDn  — layer up/down (voxel worlds)
  status     — print entity status summary
  projection — print semantic projection JSON
  log        — print narrative event log
  verbs      — list available verb templates
  goals      — show quest/goal progress
  save [f]   — save world state to file (default: savegame.json)
  load <f>   — load world state from file
  quit       — exit

[bold]Actions:[/bold] type anything you want to do.
  "move to 5,3", "attack the guard", "throw the knife",
  "speak to the guard", "intimidate the guard",
  "seduce Elara", "haggle with Tomas", …
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Semantic Sim Runtime — TUI")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--real-lm", action="store_true")
    group.add_argument("--ollama", action="store_true")
    group.add_argument("--torch", action="store_true")
    group.add_argument("--outlines", metavar="GGUF_PATH")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--torch-path",
        dest="torch_path",
        default=None,
        metavar="DIR",
        help="Local HF weights directory (use with --torch)",
    )
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--world", default="default", metavar="WORLD")
    args = parser.parse_args(argv)

    from pathlib import Path

    from .game_loop import make_test_world
    from .lm_adapter import get_adapter
    from .session_bootstrap import build_game_loop
    from .schemas import EntityKind

    world_arg = args.world
    world_path = Path(world_arg)
    if not world_path.is_absolute() and not world_path.exists():
        repo_root = Path(__file__).resolve().parent.parent.parent
        world_path = repo_root / "worlds" / world_arg
    world = make_test_world(str(world_path))

    adapter = get_adapter(
        use_real_lm=args.real_lm,
        use_ollama=args.ollama,
        use_torch=args.torch,
        use_outlines=bool(args.outlines),
        model=args.model,
        torch_model_path=args.torch_path,
        ollama_base_url=args.ollama_url,
        outlines_model_path=args.outlines,
    )
    loop = build_game_loop(world, adapter)

    player_id = None
    for eid, e in world.spatial.entities.items():
        if e.kind == EntityKind.PLAYER:
            player_id = eid
            break

    app = SimApp(loop=loop, player_id=player_id, adapter=adapter, world=world)
    app.run()


if __name__ == "__main__":
    import sys
    import warnings

    warnings.warn(
        "tui CLI merged into src.sim.play — use: python3 -m src.sim.play --tui",
        DeprecationWarning,
        stacklevel=1,
    )
    from .play import main as play_main

    play_main(["--tui", *sys.argv[1:]])
