"""
Mark-1 unified play entry — one CLI for REPL, TUI, and pygame views.

    python3 -m src.sim.play                    # terminal REPL (default)
    python3 -m src.sim.play --tui              # Textual UI
    python3 -m src.sim.play --pygame           # pygame (auto: voxel or iso)
    python3 -m src.sim.play --pygame --gl      # GPU voxel view
    python3 -m src.sim.play --pygame --view iso

Legacy flags still work: --iso, --voxel → --pygame with appropriate view.
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mark-1 Semantic Simulation — unified play client",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--tui",
        action="store_true",
        help="Textual full-screen UI",
    )
    mode.add_argument(
        "--pygame",
        action="store_true",
        help="Pygame window (iso / voxel / gl)",
    )
    # Legacy aliases → pygame + view (not in mode group so they combine with --ollama etc.)
    parser.add_argument("--iso", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--voxel", action="store_true", help=argparse.SUPPRESS)

    from .cli import (
        add_debug_args,
        add_lm_args,
        add_play_args,
        add_world_args,
    )

    add_lm_args(parser)
    add_world_args(parser)
    add_play_args(parser)
    add_debug_args(parser)
    parser.add_argument(
        "--gl",
        action="store_true",
        help="GPU OpenGL view (with --pygame; needs requirements-voxel-gl.txt)",
    )
    parser.add_argument(
        "--no-automap",
        action="store_true",
        help="REPL only: disable map after each action",
    )
    return parser


def _apply_legacy_flags(args) -> None:
    if args.iso or args.voxel or args.gl:
        args.pygame = True
    if args.gl:
        args.view = "gl"
    elif args.voxel:
        args.view = "voxel"
    elif args.iso:
        args.view = "iso"


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _apply_legacy_flags(args)

    if args.tui:
        run_tui(args)
        return

    if args.pygame:
        _run_pygame(args)
        return

    from .repl import main as repl_main
    repl_main(args=args)


def _run_pygame(args) -> None:
    from .cli import load_session, resolve_pygame_view
    from .play_client import PlayClient, run_view_only
    from .world_loader import load_world_pack

    if args.view_only:
        from .cli import resolve_world_path
        world = load_world_pack(str(resolve_world_path(args.world)))
        run_view_only(world, width=args.width, height=args.height)
        return

    session = load_session(args)
    view = resolve_pygame_view(args, session.world.spatial)
    if view == "gl":
        try:
            from .voxel_gl.renderer import gl_available
            if not gl_available():
                if getattr(args, "view", "auto") == "auto":
                    print(
                        "GL unavailable — falling back to ASCII voxel view.",
                        file=sys.stderr,
                    )
                    view = "voxel"
                else:
                    print(
                        "Install GPU deps: pip install -r requirements-voxel-gl.txt",
                        file=sys.stderr,
                    )
                    sys.exit(1)
        except ImportError:
            if getattr(args, "view", "auto") == "auto":
                view = "voxel"
            else:
                print(
                    "Install GPU deps: pip install -r requirements-voxel-gl.txt",
                    file=sys.stderr,
                )
                sys.exit(1)

    grid = session.world.spatial
    if view == "gl" and not grid.use_voxels:
        print(
            "Note: this world has no voxel volume; GL view will be empty.\n"
            "  Use: ./run_sim.sh --pygame --gl voxel_tavern",
            file=sys.stderr,
        )

    PlayClient(
        session.loop,
        view=view,
        adapter_label=session.adapter_label,
        window_w=args.width,
        window_h=args.height,
        session=session,
    ).run()


def run_tui(args) -> None:
    """TUI entry using shared session bootstrap."""
    from .cli import load_session
    from .tui import SimApp

    session = load_session(args)
    app = SimApp(
        loop=session.loop,
        player_id=session.player_id,
        adapter=session.loop.adapter,
        world=session.world,
        story_mode=session.story_mode,
    )
    app.run()


if __name__ == "__main__":
    main()
