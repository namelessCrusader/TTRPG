"""
Headless autonomous simulation runner.

All entities are driven by the LM-NPC policy; there is no player input.
Each tick: the world clock fires ambient events, then every living entity
acts once through the NPC policy.

Usage:
    python3 -m src.sim.autonomous
    python3 -m src.sim.autonomous --world worlds/tavern --ticks 80
    python3 -m src.sim.autonomous --world worlds/spicy  --ollama --model qwen3:4b
    python3 -m src.sim.autonomous --world worlds/tavern --torch --model Qwen/Qwen2.5-1.5B-Instruct
    python3 -m src.sim.autonomous --torch --torch-path /path/to/downloaded-model
    OLLAMA_MODELS=/path/to/models python3 -m src.sim.autonomous --ollama --world worlds/tavern

Flags:
    --world       PATH  world pack directory (default: worlds/tavern)
    --ticks       N     number of ticks to run (default: 60)
    --delay       SECS  pause between ticks; 0 = as fast as possible (default: 0)
    --map-every   N     redraw map every N ticks; 0 = never (default: 5)
    --ollama            use local Ollama adapter (requires `ollama serve`)
    --torch             HuggingFace + PyTorch (no Ollama server)
    --torch-path  DIR   local downloaded weights (config.json + shards)
    --model       NAME  Ollama tag or HF hub id (default: qwen3:4b)
    --mock              force mock LM adapter regardless of other flags
    --log-file    PATH  append raw event log to a file (JSONL)
    --quiet             suppress per-action narration; only show ambient + map
    --debug             per-tick state diffs (NPC meta, edges, pressures, policy branch)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _resolve_pack_path(world_arg: str) -> Path:
    p = Path(world_arg)
    if p.is_absolute():
        return p
    repo_root = Path(__file__).resolve().parent.parent.parent
    return repo_root / p


def _render_entity_action(result, world) -> str:
    """Single narrative line for one entity's action in the autonomous log."""
    from .narrator import render_event
    if result.event is None:
        return f"  {result.status}"
    narration = render_event(result.event, world)
    return f"  {result.status}\n    {narration}"


def _open_log(path: Optional[str]):
    if path is None:
        return None
    return open(path, "a", encoding="utf-8")  # noqa: WPS515


def _log_event(fh, event) -> None:
    if fh is None or event is None:
        return
    record = {
        "tick": event.tick,
        "event_id": event.event_id,
        "actor": event.action.actor,
        "verb": event.action.verb
        if isinstance(event.action.verb, str) else event.action.verb.value,
        "target": (
            event.action.target
            if isinstance(event.action.target, str)
            else (
                {"x": event.action.target.x, "y": event.action.target.y}
                if hasattr(event.action.target, "x") else None
            )
        ),
        "rationale": event.action.intent.rationale if event.action.intent else None,
        "transitions": [t.kind.value for t in event.transitions],
    }
    fh.write(json.dumps(record) + "\n")
    fh.flush()


# ---------------------------------------------------------------------------
# Pre-history simulation
# ---------------------------------------------------------------------------


def _prerun_history(world, history_ticks: int = 20) -> None:
    """
    Run the world silently through `history_ticks` ticks using a fast mock
    adapter before the main simulation begins.

    This seeds emergent world state: NPCs form goals, relationships are created,
    faction resources shift, and events accumulate in the log.  The resulting
    state is more interesting than a cold start.

    After pre-history:
    - The tick counter is reset to 0 (the main simulation begins "now")
    - The event log is trimmed to the last 15 events so NPCs have recent context
    - A short prose summary is stored in episode_memory (not entity.knowledge)
      so it does not pollute dialogue fallbacks
    """
    from .game_loop import GameLoop
    from .lm_adapter import get_adapter

    print(f"  [HISTORY] Running {history_ticks}-tick world pre-history…")
    mock_adapter = get_adapter()  # fast mock
    history_loop = GameLoop(world, mock_adapter)

    for _ in range(history_ticks):
        try:
            history_loop.autonomous_tick()
        except Exception as exc:
            logger.warning("Pre-history tick failed (best-effort): %s", exc)

    # Summarise major events into episode_memory (long-term context), not
    # entity.knowledge — knowledge is surfaced to dialogue fallbacks and must
    # stay author-authored facts from entities.yaml only.
    major_verbs = {"dialogue_spoken", "entity_goal_added", "edge_created",
                   "belief_propagated", "claim_made", "item_transferred"}
    summary_lines: list[str] = []
    for ev in world.event_log[-history_ticks:]:
        for t in ev.transitions:
            if t.kind.value in major_verbs:
                actor_id = ev.action.actor
                actor_ent = world.spatial.entities.get(actor_id)
                actor_name = actor_ent.name if actor_ent else str(actor_id)
                verb = ev.action.verb
                if isinstance(verb, str):
                    verb_str = verb
                else:
                    verb_str = verb.value.replace("_", " ")
                line = f"{actor_name} {verb_str}"
                if line not in summary_lines:
                    summary_lines.append(line)
                break
    if summary_lines:
        episode_blurb = (
            "Before the scene opened: "
            + "; ".join(summary_lines[:5])
            + "."
        )
        for entity in world.spatial.entities.values():
            if entity.kind.value == "npc":
                mem = world.episode_memory.setdefault(str(entity.entity_id), [])
                mem.append(episode_blurb)
                world.episode_memory[str(entity.entity_id)] = mem[-5:]

    # Reset world to tick 0, keep last 15 events as context
    world.tick = 0
    world.event_log = world.event_log[-15:]
    print(f"  [HISTORY] Pre-history complete. World seeded with {len(summary_lines)} events.")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def _print_debug_tick(
    world,
    before_snap: dict,
    result,
) -> None:
    from .sim_debug import capture_snapshot, format_snapshot_diff

    after = capture_snapshot(world)
    print(format_snapshot_diff(before_snap, after, title="after tick"))
    for line in getattr(result, "pressure_lines", []) or []:
        print(f"  {line}")
    for line in getattr(result, "semantic_lines", []) or []:
        print(f"  [SEMANTIC] {line}")
    for er in getattr(result, "entity_results", []) or []:
        ent = world.spatial.entities.get(er.action.actor) if er.action else None
        branch = (ent.meta.get("last_policy_branch") if ent else None) or "—"
        if ent:
            print(f"  [policy] {ent.name}: {branch}")


def run_simulation(
    world_path: Path,
    *,
    ticks: int = 60,
    delay: float = 0.0,
    map_every: int = 5,
    use_ollama: bool = False,
    use_torch: bool = False,
    torch_path: Optional[str] = None,
    model: str = "qwen3:4b",
    player_model: Optional[str] = None,
    npc_model: Optional[str] = None,
    adjudicator_model: Optional[str] = None,
    narrator_model: Optional[str] = None,
    use_mock: bool = False,
    log_file: Optional[str] = None,
    quiet: bool = False,
    save_on_exit: Optional[str] = None,
    load_path: Optional[str] = None,
    autosave_every: int = 0,
    checkpoint_dir: Optional[str] = None,
    prerun_history_ticks: int = 5,
    cognition_mode: Optional[str] = None,
    trace_dir: Optional[str] = None,
    semantic_enabled: bool = True,
    semantic_lm: bool = False,
    debug_mode: bool = False,
    session_log: Optional[str] = None,
    story_mode: bool = False,
) -> None:
    from .game_loop import GameLoop
    from .lm_adapter import MockLMAdapter, get_adapter
    from .log_tee import LogTeeSession, default_log_path
    from .npc_lm_policy import LMNpcPolicy
    from .npc_policy import ReactivePolicy
    from .narrator import render_map
    from .session_chronicle import (
        build_session_recap,
        format_action_line,
        format_beat_chapter,
        format_tick_header,
    )
    from .tracing import TraceRecorder, new_run_id
    from .world_loader import load_world_pack

    log_path = Path(session_log) if session_log else None
    if debug_mode and log_path is None:
        log_path = default_log_path("autonomous_debug")

    tee_ctx = LogTeeSession(log_path) if log_path else None
    if tee_ctx is not None:
        tee_ctx.__enter__()
        print(f"  [LOG] Writing full output to {log_path}\n")

    world = load_world_pack(world_path)
    if load_path:
        from .persistence import load_world

        world = load_world(
            load_path,
            pack_path=world_path,
            rebind_config=True,
        )
        print(f"  [LOAD] Resumed from {load_path} (tick={world.tick})")

    if use_mock:
        adapter = get_adapter()
    elif use_torch:
        adapter = get_adapter(
            use_torch=True,
            model=model,
            player_model=player_model,
            npc_model=npc_model,
            adjudicator_model=adjudicator_model,
            narrator_model=narrator_model,
            torch_model_path=torch_path,
        )
    elif use_ollama:
        adapter = get_adapter(
            use_ollama=True,
            model=model,
            player_model=player_model,
            npc_model=npc_model,
            adjudicator_model=adjudicator_model,
            narrator_model=narrator_model,
        )
    else:
        adapter = get_adapter()

    if use_torch and not use_mock:
        from .lm_adapter.torch_adapter import TorchLMAdapter

        def _torch_unavailable(ad: object) -> bool:
            if isinstance(ad, TorchLMAdapter):
                return not ad.is_ready
            inner = getattr(ad, "npc_adapter", None) or getattr(ad, "player_adapter", None)
            if isinstance(inner, TorchLMAdapter):
                return not inner.is_ready
            return False

        if _torch_unavailable(adapter):
            print(
                "\n  [WARN] Torch model failed to load — NPC/player LM inference "
                "will fall back to reactive policy each tick.\n"
                "  Fix options:\n"
                "    • BACKEND=ollama ./run_auto.sh …\n"
                "    • COGNITION=reactive_only ./run_auto.sh …\n"
                "    • TORCH_MODEL=Qwen/Qwen2.5-1.5B-Instruct (known-good default)\n"
                "    • TORCH_PATH=/path/to/hf-checkpoint (must contain config.json)\n"
                "    • GGUF files need Outlines, not --torch\n"
            )
            if cognition_mode is None:
                print("  [WARN] Continuing with LM attempts + per-tick reactive fallback.\n")

    if not semantic_enabled:
        world.config.semantic.enabled = False
    elif not use_mock and not isinstance(adapter, MockLMAdapter):
        world.config.semantic.use_mock = False
    elif semantic_lm and not use_mock:
        world.config.semantic.use_mock = False

    # ── Pre-history simulation (fast mock) ────────────────────────────────────
    if world.meta.get("magic_duel"):
        prerun_history_ticks = 0
    if prerun_history_ticks > 0 and not load_path:
        _prerun_history(world, history_ticks=prerun_history_ticks)

    if cognition_mode:
        world.config.npc_policy.cognition.mode = cognition_mode

    reactive = ReactivePolicy(
        threat_lookback_ticks=world.config.npc_policy.threat_lookback_ticks,
        flee_health_fraction=world.config.npc_policy.flee_health_fraction,
    )
    if isinstance(adapter, MockLMAdapter):
        npc_policy = reactive
    else:
        npc_policy = LMNpcPolicy(
            adapter,
            fallback=reactive,
            cognition_mode=world.config.npc_policy.cognition.mode,
            model_name=model if (use_ollama or use_torch) else "",
        )
    loop = GameLoop(world, adapter, npc_policy=npc_policy, debug_mode=debug_mode)

    trace_recorder = None
    if trace_dir:
        from pathlib import Path as _Path
        tpath = _Path(trace_dir)
        tpath.mkdir(parents=True, exist_ok=True)
        trace_recorder = TraceRecorder(
            run_id=new_run_id(),
            world_pack=str(world_path),
            seed=world.rng_seed,
            adapter_label=model if (use_ollama or use_torch) else "mock",
            policy_label=world.config.npc_policy.cognition.mode,
            output_path=tpath / f"trace_{world.rng_seed}.jsonl",
        )
        loop.trace_recorder = trace_recorder

    log_fh = _open_log(log_file)

    entity_names = [e.name for e in world.all_entities().values()]

    # ── Banner ────────────────────────────────────────────────────────────────
    if use_mock:
        model_label = "mock"
    elif use_torch:
        model_label = f"torch:{model}"
    elif use_ollama:
        model_label = model
    else:
        model_label = "mock"
    print(f"\n{'═'*60}")
    print(f"  AUTONOMOUS SIMULATION")
    print(f"  World  : {world.name}")
    print(f"  Model  : {model_label}")
    print(f"  Ticks  : {ticks}")
    print(f"  Cast   : {', '.join(entity_names)}")
    if debug_mode:
        print(f"  Debug  : ON (state diffs each tick)")
    if story_mode:
        print(f"  Story  : ON (narrative presentation)")
    if log_path is not None:
        print(f"  Log    : {log_path}")
    if checkpoint_dir:
        print(f"  Checkpoints: {checkpoint_dir}")
    if autosave_every > 0:
        print(f"  Autosave: every {autosave_every} ticks")
    print(f"{'═'*60}\n")

    if map_every > 0:
        print(render_map(world))
        print()

    # ── Tick loop ─────────────────────────────────────────────────────────────
    from .sim_debug import capture_snapshot

    policy_branches: dict[str, int] = {}

    try:
        for _ in range(ticks):
            before_snap = capture_snapshot(world) if debug_mode else None
            result = loop.autonomous_tick()

            for line in result.scenario_lines:
                if story_mode:
                    for chapter_line in format_beat_chapter(result.scenario_lines, world):
                        print(chapter_line)
                    break
                else:
                    print(f"  {line}")

            if story_mode and not result.scenario_lines:
                print(format_tick_header(world, result.tick))
                print()

            # Ambient events
            for ae in result.ambient_events:
                if ae.transitions:
                    narrative = ae.transitions[0].payload.get("narrative", "")
                    if narrative:
                        print(f"  [WORLD] tick={ae.tick}  {narrative}")
                if log_fh:
                    _log_event(log_fh, ae)

            # Entity actions
            if not quiet:
                for er in result.entity_results:
                    line = format_action_line(er, world, story_mode=story_mode)
                    if line:
                        print(line)
                    if log_fh:
                        _log_event(log_fh, er.event)
                    actor_id = er.action.actor if er.action else None
                    if actor_id:
                        ent = world.spatial.entities.get(actor_id)
                        branch = (
                            ent.meta.get("last_policy_branch") if ent else None
                        ) or "unknown"
                        policy_branches[branch] = policy_branches.get(branch, 0) + 1
                for line in result.physics_lines:
                    print(f"  [PHYSICS] {line}")

            # Completed goals
            for g in result.completed_goals:
                print(f"\n  ★ GOAL COMPLETE at tick {world.tick}: {g.title}")
                if g.reward.narrative:
                    print(f"    {g.reward.narrative}")

            for line in result.semantic_lines:
                print(f"  [SEMANTIC] {line}")

            for line in result.pressure_lines:
                if line not in result.semantic_lines:
                    print(f"  {line}")

            for line in getattr(result, "campaign_lines", []) or []:
                print(f"  {line}")

            if getattr(result, "quest_journal", None):
                from src.sim.quest_journal import format_journal_lines

                for line in format_journal_lines(result.quest_journal)[:3]:
                    print(f"  📜 {line}")

            if debug_mode and before_snap is not None:
                _print_debug_tick(world, before_snap, result)

            # Periodic map redraw
            if map_every > 0 and world.tick % map_every == 0:
                print(f"\n── tick {world.tick} ──────────────────────────────────")
                print(render_map(world))
                print()

            if delay > 0:
                time.sleep(delay)

            import sys
            sys.stdout.flush()

            if autosave_every > 0 and world.tick % autosave_every == 0:
                ck_dir = checkpoint_dir or "saves/autonomous"
                try:
                    from .persistence import save_checkpoint

                    ck_path = save_checkpoint(
                        world,
                        ck_dir,
                        pack_id=world.meta.get("pack_id") or world_path.name,
                    )
                    if not quiet:
                        print(f"  [CHECKPOINT] tick={world.tick} → {ck_path}")
                except Exception as exc:
                    print(f"  [CHECKPOINT ERROR] {exc}")

    except KeyboardInterrupt:
        print("\n[interrupted by user]")
    finally:
        if tee_ctx is not None:
            tee_ctx.__exit__(None, None, None)
            print(f"\n  [LOG] Saved session log: {log_path}")

    if trace_recorder is not None:
        trace_recorder.close(world)

    # ── Save on exit ──────────────────────────────────────────────────────────
    if save_on_exit:
        try:
            from .persistence import save_world
            n = save_world(world, save_on_exit)
            print(f"  [SAVED] {save_on_exit}  (tick={world.tick}, {n} events)")
        except Exception as exc:
            print(f"  [SAVE ERROR] {exc}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Simulation ended at tick {world.tick}")
    print(f"  Events logged : {len(world.event_log)}")
    phase = world.meta.get("scenario_phase")
    if phase is not None:
        print(f"  Scenario phase: {phase}")
    active = world.meta.get("active_pressures") or []
    if active:
        print(f"  Active pressures: {', '.join(p['id'] for p in active if p.get('id'))}")
    if policy_branches:
        branch_summary = ", ".join(
            f"{k}={v}" for k, v in sorted(policy_branches.items(), key=lambda x: -x[1])
        )
        print(f"  Policy branches: {branch_summary}")
    from .run_metrics import format_run_metrics

    for line in format_run_metrics(world):
        print(line)
    if story_mode:
        for line in build_session_recap(world):
            print(line)
    print(f"{'─'*60}\n")

    if log_fh:
        log_fh.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python3 -m src.sim.autonomous",
        description="Run a headless autonomous simulation (no player input).",
    )
    parser.add_argument(
        "--world",
        default="worlds/tavern",
        metavar="PATH",
        help="World pack directory (default: worlds/tavern)",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=60,
        metavar="N",
        help="Number of ticks to simulate (default: 60)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        metavar="SECS",
        help="Pause between ticks in seconds (default: 0 = as fast as possible)",
    )
    parser.add_argument(
        "--map-every",
        type=int,
        default=5,
        dest="map_every",
        metavar="N",
        help="Redraw map every N ticks; 0 = never (default: 5)",
    )
    lm_group = parser.add_mutually_exclusive_group()
    lm_group.add_argument(
        "--ollama",
        action="store_true",
        help="Use local Ollama adapter (requires `ollama serve`)",
    )
    lm_group.add_argument(
        "--torch",
        action="store_true",
        help="Use HuggingFace transformers + PyTorch (hub id or local path)",
    )
    parser.add_argument(
        "--model",
        default="qwen3:4b",
        metavar="NAME",
        help="Ollama model tag, or HF hub id with --torch (default: qwen3:4b)",
    )
    parser.add_argument(
        "--player-model",
        dest="player_model",
        default=None,
        metavar="NAME",
        help="LM for player-facing infer/adjudicate (defaults to --model)",
    )
    parser.add_argument(
        "--npc-model",
        dest="npc_model",
        default=None,
        metavar="NAME",
        help="LM for NPC infer/replies (defaults to --model)",
    )
    parser.add_argument(
        "--adjudicator-model",
        dest="adjudicator_model",
        default=None,
        metavar="NAME",
        help=(
            "LM for adjudicate()/propose_consequences() only. "
            "Pick a small schema-tight model so rescued actions "
            "resolve reproducibly (defaults to --player-model)."
        ),
    )
    parser.add_argument(
        "--narrator-model",
        dest="narrator_model",
        default=None,
        metavar="NAME",
        help=(
            "LM for narrate() only (free prose, no state mutation). "
            "Pick a model tuned for evocative prose; the firewall "
            "prevents narration from leaking into mechanics, so this "
            "is variance-free (defaults to --npc-model)."
        ),
    )
    parser.add_argument(
        "--torch-path",
        dest="torch_path",
        default=None,
        metavar="DIR",
        help=(
            "Local HF weights directory (config.json + shards). "
            "Use with --torch; overrides --model for loading."
        ),
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock LM adapter (fast, deterministic, no network)",
    )
    parser.add_argument(
        "--log-file",
        metavar="PATH",
        help="Append raw event JSONL to this file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-action narration; only show ambient events and map",
    )
    parser.add_argument(
        "--save-on-exit",
        metavar="PATH",
        help="Save world state to this file when the simulation ends",
    )
    parser.add_argument(
        "--load",
        metavar="PATH",
        default=None,
        help="Resume from a savegame JSON (world pack still sets config)",
    )
    parser.add_argument(
        "--autosave-every",
        type=int,
        default=0,
        metavar="N",
        help="Write a rotating checkpoint every N ticks (0 = disabled)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        metavar="PATH",
        default=None,
        help="Directory for rotating checkpoints (default: saves/autonomous)",
    )
    parser.add_argument(
        "--history-ticks",
        type=int,
        default=15,
        dest="history_ticks",
        metavar="N",
        help="Ticks of silent pre-history to run before the main simulation (default: 15, 0 = skip)",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable delayed semantic dynamics pass",
    )
    parser.add_argument(
        "--semantic-lm",
        action="store_true",
        default=None,
        dest="semantic_lm",
        help="Use the main LM for semantic pressure (default: on with ollama/torch)",
    )
    parser.add_argument(
        "--no-semantic-lm",
        action="store_false",
        dest="semantic_lm",
        help="Force rule-based semantic pressure even with a real LM",
    )
    parser.add_argument(
        "--cognition",
        default=None,
        metavar="MODE",
        help="NPC cognition: lm (default) | reactive_only",
    )
    parser.add_argument(
        "--trace-dir",
        default=None,
        metavar="PATH",
        help="Write research trace JSONL to this directory",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-tick state diffs (NPC meta, edges, pressures, policy branch)",
    )
    parser.add_argument(
        "--story",
        action="store_true",
        help="Story-mode output: chapter breaks, filtered waits, session recap",
    )
    parser.add_argument(
        "--session-log",
        metavar="PATH",
        default=None,
        help="Mirror all stdout to this file (default: logs/autonomous_debug_<time>.log when --debug)",
    )

    args = parser.parse_args(argv)

    run_simulation(
        _resolve_pack_path(args.world),
        ticks=args.ticks,
        delay=args.delay,
        map_every=args.map_every,
        use_ollama=args.ollama,
        use_torch=args.torch,
        torch_path=args.torch_path,
        model=args.model,
        player_model=args.player_model,
        npc_model=args.npc_model,
        adjudicator_model=args.adjudicator_model,
        narrator_model=args.narrator_model,
        use_mock=args.mock,
        log_file=args.log_file,
        quiet=args.quiet,
        save_on_exit=args.save_on_exit,
        load_path=args.load,
        autosave_every=args.autosave_every,
        checkpoint_dir=args.checkpoint_dir,
        prerun_history_ticks=args.history_ticks,
        cognition_mode=args.cognition,
        trace_dir=args.trace_dir,
        semantic_enabled=not args.no_semantic,
        semantic_lm=args.semantic_lm,
        debug_mode=args.debug,
        session_log=args.session_log,
        story_mode=args.story,
    )


if __name__ == "__main__":
    main()
