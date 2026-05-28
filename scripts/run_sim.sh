#!/usr/bin/env bash
# Interactive play — single entry (REPL, TUI, or pygame).
#
#   ./run_sim.sh                    — terminal REPL (default world: spicy)
#   ./run_sim.sh tavern             — REPL, named world
#   ./run_sim.sh --tui tavern       — Textual UI
#   ./run_sim.sh --pygame voxel_tavern   — pygame (auto view: voxel or iso)
#   ./run_sim.sh --pygame --gl voxel_tavern   — GPU 3D (pip install -r requirements-voxel-gl.txt)
#   ./run_sim.sh --story tavern     — narrative chapter breaks + scene banner
#   ./run_sim.sh --debug tavern     — REPL with state diffs
#
# Legacy aliases (still work): --iso, --voxel → same as --pygame
#
# Headless: ./run_auto.sh

WORLD="castle"
USE_TUI=0
USE_PYGAME=0
USE_GL=0
DEBUG=0
EXPLICIT_LM=0
EXTRA_ARGS=()
NEXT_IS_WORLD=0
for arg in "$@"; do
  if [ "$NEXT_IS_WORLD" = "1" ]; then
    WORLD="$arg"
    NEXT_IS_WORLD=0
    continue
  fi
  if [ "$arg" = "--tui" ]; then
    USE_TUI=1
  elif [ "$arg" = "--pygame" ] || [ "$arg" = "--iso" ] || [ "$arg" = "--voxel" ]; then
    USE_PYGAME=1
    EXTRA_ARGS+=("$arg")
  elif [ "$arg" = "--gl" ]; then
    USE_GL=1
    EXTRA_ARGS+=("$arg")
  elif [ "$arg" = "--debug" ]; then
    DEBUG=1
    EXTRA_ARGS+=(--debug)
  elif [ "$arg" = "--mock" ] || [ "$arg" = "--ollama" ] || [ "$arg" = "--torch" ] || [ "$arg" = "--real-lm" ]; then
    EXPLICIT_LM=1
    EXTRA_ARGS+=("$arg")
  elif [ "$arg" = "--world" ]; then
    NEXT_IS_WORLD=1
  elif [[ "$arg" == --* ]]; then
    EXTRA_ARGS+=("$arg")
  else
    WORLD="$arg"
  fi
done

OLLAMA_MODELS="/home/nisargparikh/Desktop/Fun Stuff/Mark_1/models"
export OLLAMA_MODELS

LM_ARGS=()
if [ "$EXPLICIT_LM" = "0" ]; then
  if [[ "$BACKEND" == "torch" ]]; then
    if [[ -n "${TORCH_PATH:-}" ]]; then
      LM_ARGS=(--torch --torch-path "$TORCH_PATH")
    else
      LM_ARGS=(--torch --model "$TORCH_MODEL")
    fi
  else
    LM_ARGS=(--ollama --model "${MODEL:-qwen3:1.7b}")
  fi
fi

PLAY_ARGS=()
if [ "$USE_TUI" = "1" ]; then
  PLAY_ARGS=(--tui)
elif [ "$USE_PYGAME" = "1" ]; then
  PLAY_ARGS=(--pygame)
fi
if [ "$USE_GL" = "1" ]; then
  PLAY_ARGS+=(--gl)
fi

python3 -m src.sim.play "${PLAY_ARGS[@]}" "${LM_ARGS[@]}" --world "$WORLD" "${EXTRA_ARGS[@]}"
