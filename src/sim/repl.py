"""
Interactive REPL for the semantic simulation runtime.

Run with:
    python3 -m src.sim.repl                          (mock LM, no network)
    python3 -m src.sim.repl --ollama                 (local Ollama, default qwen3:4b)
    python3 -m src.sim.repl --ollama --model qwen3:8b
    python3 -m src.sim.repl --torch --model Qwen/Qwen2.5-1.5B-Instruct
    python3 -m src.sim.repl --torch --torch-path /path/to/downloaded-model
    python3 -m src.sim.repl --real-lm                (OpenAI, requires OPENAI_API_KEY)
    python3 -m src.sim.repl --outlines /path/to.gguf (llama.cpp, Level 3 constrained)
"""

from __future__ import annotations

import argparse
import json

from .game_loop import make_test_world
from .lm_adapter import get_adapter
from .session_bootstrap import build_game_loop
from .narrator import render_event, render_event_log, render_inventory, render_map, render_world_summary
from .projection import project
from .schemas import EntityKind


def _player_id(world):
    for eid, e in world.spatial.entities.items():
        if e.kind == EntityKind.PLAYER:
            return eid
    return None


def _print_lm_debug(adapter, result) -> None:
    """Render the raw LM JSON (if the adapter recorded one) and the
    parsed verb / target / manner / effects extracted from it.

    The point is to diagnose cases where the LM is picking an unexpected
    verb (e.g. mapping "Jump to the guard" to verb="speak"). Showing the
    raw output reveals whether the bias lives in the prompt, the model,
    or the parser."""
    raw = getattr(adapter, "last_raw_response", None)
    action = result.action if result and getattr(result, "action", None) else None
    print()
    if raw is not None:
        latency = getattr(adapter, "last_latency_ms", None)
        if latency is not None:
            print(f"  [LM RAW {latency:.0f}ms] {raw[:400]}"
                  + ("..." if len(raw) > 400 else ""))
        else:
            print(f"  [LM RAW] {raw[:400]}"
                  + ("..." if len(raw) > 400 else ""))
    if action is not None:
        target_repr = action.target if action.target is not None else "null"
        manner = (action.intent.manner if action.intent else None) or ""
        proposed = [
            f"{p.kind}({p.payload})" for p in action.proposed_effects
        ]
        contest = (
            f"{action.contest.actor_attribute} vs "
            f"{action.contest.target_attribute or '∅'}"
            if action.contest else "none"
        )
        print(f"  [PARSED] verb={action.verb!r} target={target_repr!r} "
              f"manner={manner!r} contest={contest} "
              f"proposed_effects={proposed}")


_HELP = """\
Commands:
  map                  — draw the ASCII grid map (fog of war applied)
  status               — print entity status summary
  inventory / inv / i  — print carry capacity, equipped slots, bags and contents
  projection           — print current semantic projection (JSON)
  log                  — print narrative event log
  verbs                — print verbs that have pack-declared templates
  save [file]          — save current world state to file (default: savegame.json)
  load <file>          — load world state from file and resume
  checkpoint           — write rotating checkpoint (needs --checkpoint-dir)
  chronicle            — show player story so far
  metrics / lm         — LM call counts and latency
  stream [file]        — export presentation frames JSON
  goals                — print quest/goal progress
  debug                — toggle per-turn state diff (or use --debug on launch)
  state [name]         — dump NPC internal meta, memory, edges
  world                — world meta, active pressures, scenario phase
  layer / layer N      — Z-slice map (voxel worlds; DF-style height)
  layer up / layer down — change slice height
  quit                 — exit

Speech (best for NPC reactions — use exact words):
  say to Mira: What was that noise?
  tell Ser Aldric: Stand down.
  whisper to Linna: Meet me outside.

The action vocabulary is OPEN — type whatever verb fits what you want
to do. Examples that exercise the freeform path:

  Kernel verbs (have bespoke physics):
    "move to 5,3", "attack the guard", "throw the knife at the guard",
    "give the knife to the guard", "take the throwing knife"

  Conventional non-kernel verbs (built-in defaults):
    "intimidate the guard", "persuade the guard", "ask the guard about
    the alarm", "observe the guard", "flee", "hide", "wait", "embrace
    the guard"

  Freeform verbs (engine resolves via pack template or pure freeform):
    "console the guard about her dead brother"
    "tease the guard about his mustache"
    "haggle with the guard over the price of passage"
    "compliment the guard's polished helm"
    "sing a tavern ballad about three sailors"
    "compose a poem about the moon"
    "court the guard"

If you don't see the verb you wanted in the parsed action, run with
--debug-lm and we'll print the raw LM JSON next to the parsed verb.
"""


def main(argv=None, *, args=None):
    if args is None:
        parser = argparse.ArgumentParser(description="Semantic Sim Runtime REPL")
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            "--real-lm", action="store_true",
            help="OpenAI Structured Outputs (Level 2 constrained, requires OPENAI_API_KEY)",
        )
        group.add_argument(
            "--ollama", action="store_true",
            help="Local Ollama adapter (Level 2 constrained). Default model: qwen3:4b",
        )
        group.add_argument(
            "--torch", action="store_true",
            help="HuggingFace transformers + PyTorch (hub id or local path). Default: Qwen2.5-1.5B-Instruct",
        )
        group.add_argument(
            "--outlines", metavar="GGUF_PATH",
            help="Outlines + llama.cpp (Level 3 logit-mask constrained). Path to GGUF file.",
        )
        parser.add_argument("--model", default=None, help="Override model name or HF hub id")
        parser.add_argument(
            "--torch-path",
            dest="torch_path",
            default=None,
            metavar="DIR",
            help="Local HF weights directory (use with --torch)",
        )
        parser.add_argument(
            "--ollama-url", default="http://localhost:11434",
            help="Ollama server URL (default: http://localhost:11434)",
        )
        parser.add_argument(
            "--no-automap", action="store_true",
            help="Disable automatic map display after each action",
        )
        parser.add_argument(
            "--debug-lm", action="store_true",
            help=(
                "After each turn, print the raw LM JSON response and the "
                "verb/manner/effects the parser extracted. Useful when the LM "
                "appears to be picking the wrong verb."
            ),
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help=(
                "After each turn, print state diffs (NPC meta, edges, pressures) "
                "and policy branches. Same as the 'debug' command."
            ),
        )
        parser.add_argument(
            "--story",
            action="store_true",
            help="Story-mode presentation: chapter breaks, tension bar, filtered waits",
        )
        parser.add_argument(
            "--world", default="default", metavar="WORLD",
            help=(
                "World pack to load. Either a name under worlds/ "
                "(e.g. 'spicy', 'tavern') or a full path. "
                "Default: 'default'"
            ),
        )
        args = parser.parse_args(argv)

    # Resolve --world: bare names become worlds/<name>
    from pathlib import Path
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
    loop = build_game_loop(world, adapter, debug_mode=args.debug)
    pid = _player_id(world)
    from .cli import PlaySession

    session = PlaySession(
        world=world,
        loop=loop,
        player_id=pid,
        adapter_label="repl",
        world_path=world_path,
        adapter=adapter,
        story_mode=getattr(args, "story", False),
    )
    automap = not args.no_automap
    from .sim_debug import capture_snapshot, format_snapshot_diff, format_entity_state, format_world_debug, is_debug_mode, set_debug_mode

    if args.real_lm:
        adapter_label = f"OpenAI Structured Outputs ({args.model or 'gpt-4o-mini'}) [Level 2]"
    elif args.ollama:
        adapter_label = f"Ollama ({args.model or 'qwen3:4b'}) [Level 2 constrained]"
    elif args.torch:
        from .lm_adapter import TorchLMAdapter
        adapter_label = f"Torch/HF ({args.model or TorchLMAdapter.DEFAULT_MODEL}) [Level 1 parse]"
    elif args.outlines:
        adapter_label = f"Outlines+llama.cpp ({args.outlines}) [Level 3 logit-mask]"
    else:
        adapter_label = "Mock (deterministic, no LM)"

    from .voxel_layers import LayerNavigator, render_layer_ascii, layer_help_lines

    _pz = 0
    if pid and pid in world.spatial.entities:
        _pz = world.spatial.entities[pid].position.z
    layer_nav = LayerNavigator(
        depth=max(1, world.spatial.depth),
        slice_z=_pz if world.spatial.depth > 1 else None,
    )

    print(f"\n=== Semantic Simulation Runtime ===")
    print(f"Adapter : {adapter_label}")
    print(f"World   : {world.name}")
    if args.story:
        print(f"Story   : ON (narrative presentation)")
    if world.spatial.use_voxels or world.spatial.depth > 1:
        print(f"Volume  : {world.spatial.width}×{world.spatial.height}×{world.spatial.depth}  "
              f"({layer_nav.label()})")
        print("          Use: layer, layer N, layer up/down  (or ./run_sim.sh --voxel)")
    print(f"Map     : auto {'on' if automap else 'off'}  (toggle with --no-automap)\n")

    from .session_chronicle import format_scene_opening

    for line in format_scene_opening(world):
        print(line)

    if world.spatial.use_voxels and world.spatial.depth > 1 and pid:
        z = layer_nav.slice_z if layer_nav.slice_z is not None else world.spatial.entities[pid].position.z
        print(render_layer_ascii(world.spatial, z, player_id=pid))
    else:
        print(render_map(world, pid))
    print()

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not raw:
            continue

        if raw.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        if raw.lower() == "help":
            print(_HELP)
            if world.spatial.use_voxels or world.spatial.depth > 1:
                for line in layer_help_lines():
                    print(line)
            continue

        layer_msg = layer_nav.handle_repl_command(raw)
        if layer_msg is not None:
            print(f"Layer: {layer_msg}")
            if layer_nav.is_slice_mode and layer_nav.slice_z is not None:
                print(render_layer_ascii(world.spatial, layer_nav.slice_z, player_id=pid))
            print()
            continue

        if raw.lower() in ("map", "m"):
            if world.spatial.use_voxels and world.spatial.depth > 1:
                z = (
                    layer_nav.slice_z
                    if layer_nav.is_slice_mode
                    else world.spatial.entities[pid].position.z
                )
                print(render_layer_ascii(world.spatial, z, player_id=pid))
            else:
                print(render_map(world, pid))
            print()
            continue

        if raw.lower() == "status":
            print(render_world_summary(world))
            print()
            continue

        if raw.lower() in ("inventory", "inv", "i"):
            from .schemas import EntityKind as _EK
            player_ent = world.spatial.entities.get(pid)
            if player_ent:
                print(render_inventory(player_ent, world))
            else:
                print("  (no player entity)")
            print()
            continue

        if raw.lower() == "projection":
            proj = project(world, pid)
            print(json.dumps(json.loads(proj.model_dump_json()), indent=2))
            print()
            continue

        if raw.lower() == "log":
            lines = render_event_log(world)
            if lines:
                for line in lines:
                    print(line)
            else:
                print("(no events yet)")
            print()
            continue

        if raw.lower() == "verbs":
            print("  Kernel verbs (bespoke physics):")
            for v in sorted(["move", "attack", "take", "give", "throw"]):
                print(f"    {v}")
            print("  Pack-declared verb templates:")
            for v in sorted(world.config.verb_templates.keys()):
                tmpl = world.config.verb_templates[v]
                contest = (
                    f" (contest: {tmpl.contest.actor_attribute} vs "
                    f"{tmpl.contest.target_attribute or 'difficulty'})"
                    if tmpl.contest else ""
                )
                print(f"    {v}{contest}")
            print(
                "  Any other verb is legal too — the engine uses built-in "
                "defaults for conventional verbs (intimidate, observe, "
                "speak, etc.) and a pure freeform path for everything else."
            )
            print()
            continue

        lower = raw.lower().strip()

        from .play_commands import handle_persistence_command

        def _emit(msg: str, color: str = "normal") -> None:
            print(f"  {msg}")

        if handle_persistence_command(
            raw, session=session, world=world, loop=loop, emit=_emit,
        ):
            world = session.world
            loop = session.loop
            pid = session.player_id
            if lower.startswith("load"):
                print(render_map(world, pid))
            print()
            continue

        if lower == "goals":
            from .quest_journal import build_quest_journal, format_journal_lines

            entries = build_quest_journal(world, pid)
            if not entries:
                print("  (no active objectives or concerns)")
            else:
                for line in format_journal_lines(entries):
                    print(f"  {line}")
            print()
            continue

        if lower == "debug":
            set_debug_mode(world, not is_debug_mode(world))
            print(f"  Debug mode: {'ON' if is_debug_mode(world) else 'OFF'}\n")
            continue

        if lower == "world":
            print(format_world_debug(world))
            print()
            continue

        if lower.startswith("state"):
            parts = raw.split(None, 1)
            if len(parts) > 1:
                print(format_entity_state(world, parts[1]))
            else:
                for eid, ent in world.spatial.entities.items():
                    if ent.kind == EntityKind.NPC:
                        print(format_entity_state(world, ent.name))
            print()
            continue

        # --- process action ---
        before_snap = capture_snapshot(world) if is_debug_mode(world) else None
        result = loop.step(raw)
        from .play_commands import _maybe_autosave, apply_step_result

        _maybe_autosave(session)

        if args.debug_lm:
            _print_lm_debug(adapter, result)

        def _emit(msg: str, color: str = "normal") -> None:
            if msg:
                print(msg)

        print()
        apply_step_result(result, world, _emit, story_mode=args.story)

        if is_debug_mode(world):
            if before_snap is not None:
                print()
                print(format_snapshot_diff(before_snap, capture_snapshot(world), title="after your turn"))
            for nr in result.npc_results:
                ent = world.spatial.entities.get(nr.action.actor) if nr.action else None
                if ent:
                    branch = ent.meta.get("last_policy_branch", "—")
                    print(f"  [policy] {ent.name}: {branch}")

        print()

        # Auto-render map after every action so the player sees movement
        if automap:
            print(render_map(world, pid))
            print()


if __name__ == "__main__":
    import sys
    import warnings

    if __package__ and "play" not in sys.modules:
        pass  # direct: python -m src.sim.repl
    warnings.warn(
        "repl CLI also available as: python3 -m src.sim.play",
        DeprecationWarning,
        stacklevel=1,
    )
    main()
