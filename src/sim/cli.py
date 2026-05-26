"""
Shared CLI: world loading, LM adapters, GameLoop bootstrap.

All interactive front-ends (REPL, TUI, pygame) use this module.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class PlaySession:
    """Loaded world + game loop ready for a front-end."""

    world: object
    loop: object
    player_id: Optional[str]
    adapter_label: str
    world_path: Path
    checkpoint_dir: Optional[Path] = None
    autosave_every: int = 0  # 0 = disabled
    save_path: Optional[Path] = None
    adapter: object = None
    story_mode: bool = False


def recreate_loop(session: PlaySession, world: object) -> object:
    """Rebuild GameLoop after load while preserving session settings."""
    from .game_loop import GameLoop
    from .lm_budget import wrap_adapter_metrics

    old_loop = session.loop
    loop = GameLoop(
        world,
        adapter=session.adapter or getattr(old_loop, "adapter", None),
        player_id=getattr(old_loop, "player_id", session.player_id),
        npc_policy=getattr(old_loop, "npc_policy", None),
        npcs_act_each_turn=getattr(old_loop, "npcs_act_each_turn", True),
        player_policy=getattr(old_loop, "player_policy", None),
        debug_mode=getattr(old_loop, "debug_mode", False),
        emit_presentation=getattr(old_loop, "emit_presentation", True),
    )
    if session.adapter:
        wrap_adapter_metrics(session.adapter, world)
    session.world = world
    session.loop = loop
    from .schemas import EntityKind

    for eid, ent in world.spatial.entities.items():
        if ent.kind == EntityKind.PLAYER:
            session.player_id = eid
            loop.player_id = eid
            break
    return loop


def add_lm_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--real-lm", action="store_true")
    group.add_argument("--ollama", action="store_true")
    group.add_argument("--torch", action="store_true")
    group.add_argument("--mock", action="store_true")
    group.add_argument("--outlines", metavar="GGUF_PATH")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--player-model",
        dest="player_model",
        default=None,
        metavar="NAME",
        help="LM for player infer/adjudicate (defaults to --model)",
    )
    parser.add_argument(
        "--npc-model",
        dest="npc_model",
        default=None,
        metavar="NAME",
        help="LM for NPC infer/replies (defaults to --model)",
    )
    parser.add_argument(
        "--torch-path",
        dest="torch_path",
        default=None,
        metavar="DIR",
    )
    parser.add_argument("--ollama-url", default="http://localhost:11434")


def add_world_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--world",
        default="default",
        metavar="WORLD",
        help="World pack name under worlds/ or path",
    )


def add_play_args(parser: argparse.ArgumentParser) -> None:
    """View / window flags for pygame play mode."""
    parser.add_argument(
        "--reactive-npcs",
        action="store_true",
        help="Force deterministic reactive NPCs even when using a real LM",
    )
    parser.add_argument(
        "--view",
        choices=("auto", "iso", "voxel", "gl"),
        default="auto",
        help="Pygame renderer: auto picks from world (voxel→iso stack, --gl for GPU)",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument(
        "--view-only",
        action="store_true",
        help="Map viewer only (no LM / no commands)",
    )
    parser.add_argument(
        "--load",
        metavar="SAVE",
        default=None,
        help="Resume from a savegame JSON file",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        metavar="DIR",
        help="Directory for rotating autosave checkpoints",
    )
    parser.add_argument(
        "--autosave-every",
        type=int,
        default=0,
        metavar="N",
        help="Autosave checkpoint every N ticks (requires --checkpoint-dir)",
    )
    parser.add_argument(
        "--rebind-pack",
        action="store_true",
        help="When loading a save, refresh world.config from the pack on disk",
    )
    parser.add_argument(
        "--story",
        action="store_true",
        help="Story-mode presentation: chapter breaks, tension bar, filtered waits",
    )


def add_debug_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--debug", action="store_true", help="Per-turn state diffs")
    parser.add_argument("--debug-lm", action="store_true", help="Print raw LM JSON")


def resolve_world_path(world_arg: str) -> Path:
    world_path = Path(world_arg)
    if not world_path.is_absolute() and not world_path.exists():
        world_path = _REPO_ROOT / "worlds" / world_arg
    if not world_path.exists():
        world_path = Path(world_arg)
    return world_path


def adapter_label(args) -> str:
    if getattr(args, "mock", False):
        return "Mock"
    if args.real_lm:
        return f"OpenAI ({args.model or 'gpt-4o-mini'})"
    if args.ollama:
        return f"Ollama ({args.model or 'qwen3:4b'})"
    if args.torch:
        from .lm_adapter import TorchLMAdapter
        return f"Torch ({args.model or TorchLMAdapter.DEFAULT_MODEL})"
    if getattr(args, "outlines", None):
        return f"Outlines ({args.outlines})"
    return "Mock (default)"


def resolve_pygame_view(args, grid) -> str:
    """Choose iso | voxel | gl from flags and world."""
    if getattr(args, "view", "auto") != "auto":
        return args.view
    if getattr(args, "gl", False):
        return "gl"
    if getattr(args, "voxel", False):
        return "voxel"
    if getattr(args, "iso", False):
        return "iso"
    if grid.use_voxels or grid.depth > 1:
        try:
            from .voxel_gl.renderer import gl_available
            if gl_available():
                return "gl"
        except ImportError:
            pass
        return "voxel"
    return "iso"


def load_session(
    args,
    *,
    repl_mode: bool = False,
) -> PlaySession:
    """Load world pack, adapter, and GameLoop with unified NPC policy."""
    from .schemas import EntityKind

    world_path = resolve_world_path(args.world)
    load_path = getattr(args, "load", None)
    rebind = getattr(args, "rebind_pack", False)

    if load_path:
        from .persistence import load_world

        world = load_world(
            load_path,
            pack_path=str(world_path) if rebind else None,
            rebind_config=rebind,
        )
    elif repl_mode:
        from .game_loop import make_test_world
        world = make_test_world(str(world_path))
    else:
        from .world_loader import load_world_pack
        world = load_world_pack(str(world_path))

    from .lm_adapter import get_adapter, MockLMAdapter

    adapter = get_adapter(
        use_real_lm=args.real_lm,
        use_ollama=args.ollama,
        use_torch=args.torch,
        use_outlines=bool(getattr(args, "outlines", None)),
        model=args.model,
        player_model=getattr(args, "player_model", None),
        npc_model=getattr(args, "npc_model", None),
        torch_model_path=getattr(args, "torch_path", None),
        ollama_base_url=args.ollama_url,
        outlines_model_path=getattr(args, "outlines", None),
    )
    if getattr(args, "mock", False):
        adapter = MockLMAdapter()

    if not world.meta.get("pack_id"):
        world.meta["pack_id"] = world_path.name

    debug = getattr(args, "debug", False)
    force_reactive = getattr(args, "reactive_npcs", False)
    from .session_bootstrap import build_game_loop

    loop = build_game_loop(
        world,
        adapter,
        force_reactive_npcs=force_reactive,
        debug_mode=debug,
    )

    player_id = None
    for eid, ent in world.spatial.entities.items():
        if ent.kind == EntityKind.PLAYER:
            player_id = eid
            break

    ckpt = getattr(args, "checkpoint_dir", None)
    autosave = int(getattr(args, "autosave_every", 0) or 0)

    return PlaySession(
        world=world,
        loop=loop,
        player_id=player_id,
        adapter_label=adapter_label(args),
        world_path=world_path,
        checkpoint_dir=Path(ckpt) if ckpt else None,
        autosave_every=autosave if ckpt else 0,
        save_path=Path(load_path) if load_path else None,
        adapter=adapter,
        story_mode=getattr(args, "story", False),
    )
