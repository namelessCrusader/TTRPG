#!/usr/bin/env bash
# Autonomous/headless simulation — no player, all NPCs driven by the LM.
#
# Usage:
#   ./run_auto.sh                        — voxel_tavern world, 10 ticks, Ollama
#   ./run_auto.sh spicy                  — spicy world, 60 ticks
#   ./run_auto.sh <world_dir> [ticks]    — any world, optional tick count
#   ./run_auto.sh magic_duel 60         — 1v1 spell duel (test magic systems)
#   COGNITION=reactive_only ./run_auto.sh tavern 60  — deterministic, no LM
#   ./run_auto.sh tavern 120 --delay 1   — 120 ticks, 1-second pause between
#   WORLD=spicy TICKS=40 ./run_auto.sh   — env-var alternative
#
# Tiered models (player infer vs NPC cognition):
#   PLAYER_MODEL=qwen2.5:7b NPC_MODEL=gemma3:1b ./run_auto.sh voxel_tavern 30
#
# Role-tiered models (split adjudicator + narrator off the player path):
#   PLAYER_MODEL=qwen2.5:7b NPC_MODEL=gemma3:1b \
#   ADJUDICATOR_MODEL=qwen2.5:3b NARRATOR_MODEL=qwen2.5:7b \
#     ./run_auto.sh voxel_tavern 30
# Pick a small schema-tight model for ADJUDICATOR_MODEL (reproducibility wins
# over poetic ruling text); pick a richer prose model for NARRATOR_MODEL —
# the firewall blocks narration from mutating state so this costs nothing
# in mechanics.
#
# Extra flags are passed directly to autonomous.py (--quiet, --map-every, --debug, etc.)
#   ./run_auto.sh tavern 60 --quiet --log-file sim.jsonl
#   ./run_auto.sh --debug             — default voxel_tavern; flags-only also works
#   ./run_auto.sh tavern 30 --debug   — per-tick state diffs + logs/autonomous_debug_<time>.log
#   Default cognition is lm (structured infer_npc each tick for every model).
#   ./run_auto.sh tavern 60 --story          — narrative chapter breaks + recap
#   Semantic LM pressure is ON by default with ollama/torch (--no-semantic-lm to disable)
#   BACKEND=torch TORCH_MODEL=Qwen/Qwen2.5-1.5B-Instruct ./run_auto.sh tavern 30
#   ./run_auto.sh tavern 30 --torch --torch-path /path/to/downloaded-hf-model
#   BACKEND=torch TORCH_PATH=/path/to/downloaded-hf-model ./run_auto.sh tavern 30

WORLD="${WORLD:-voxel_tavern}"
TICKS="${TICKS:-10}"
PASSTHRU=()
while [[ $# -gt 0 ]]; do
  if [[ "$1" == -* ]]; then
    PASSTHRU+=("$1")
    shift
  elif [[ -z "${_WORLD_ARG:-}" ]]; then
    WORLD="$1"
    _WORLD_ARG=1
    shift
  elif [[ -z "${_TICKS_ARG:-}" && "$1" =~ ^[0-9]+$ ]]; then
    TICKS="$1"
    _TICKS_ARG=1
    shift
  else
    PASSTHRU+=("$1")
    shift
  fi
done

COGNITION="${COGNITION:-}"
TRACE_DIR="${TRACE_DIR:-}"

EXTRA_ARGS=()
if [[ -n "$COGNITION" ]]; then
  EXTRA_ARGS+=(--cognition "$COGNITION")
fi
if [[ -n "$TRACE_DIR" ]]; then
  EXTRA_ARGS+=(--trace-dir "$TRACE_DIR")
fi

BACKEND="${BACKEND:-torch}"
TORCH_MODEL="${TORCH_MODEL:-Qwen/Qwen3.5-9B}"
# TORCH_MODEL="${TORCH_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
# TORCH_MODEL="${TORCH_MODEL:-DavidAU/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking}"
# OLLAMA_MODEL="${OLLAMA_MODEL:-dolphin-phi:latest}"
OLLAMA_MODEL="${OLLAMA_MODEL:-gemma3:1b}"
# Stronger default for player intent parsing when tiered models are enabled
PLAYER_MODEL="${PLAYER_MODEL:-Qwen/Qwen3.5-9B}"
# PLAYER_MODEL="${PLAYER_MODEL:-qwen3:1.7b}"
NPC_MODEL="${NPC_MODEL:-$TORCH_MODEL}"
# NPC_MODEL="${NPC_MODEL:-$OLLAMA_MODEL}"
# Role-tiered slots — empty = inherit from player/npc model.
ADJUDICATOR_MODEL="${ADJUDICATOR_MODEL:-}"
NARRATOR_MODEL="${NARRATOR_MODEL:-}"

LM_ARGS=()
if [[ "$BACKEND" == "torch" ]]; then
  if [[ -n "${TORCH_PATH:-}" ]]; then
    LM_ARGS=(--torch --torch-path "$TORCH_PATH")
    if [[ -n "${PLAYER_TORCH_PATH:-}" ]]; then
      LM_ARGS+=(--player-model "$PLAYER_TORCH_PATH" --npc-model "$TORCH_PATH")
    elif [[ -n "${PLAYER_MODEL:-}" && "$PLAYER_MODEL" != "$TORCH_MODEL" ]]; then
      LM_ARGS+=(--player-model "$PLAYER_MODEL" --npc-model "${NPC_MODEL:-$TORCH_MODEL}")
    else
      LM_ARGS+=(--model "$TORCH_MODEL")
    fi
  else
    LM_ARGS=(--torch --model "$TORCH_MODEL")
    if [[ -n "${PLAYER_MODEL:-}" && "$PLAYER_MODEL" != "$TORCH_MODEL" ]]; then
      LM_ARGS+=(--player-model "$PLAYER_MODEL" --npc-model "${NPC_MODEL:-$TORCH_MODEL}")
    fi
  fi
else
  LM_ARGS=(--ollama --model "$OLLAMA_MODEL")
  if [[ -n "${PLAYER_MODEL:-}" && "$PLAYER_MODEL" != "$OLLAMA_MODEL" ]]; then
    LM_ARGS+=(--player-model "$PLAYER_MODEL" --npc-model "$NPC_MODEL")
  fi
fi

# Role-tier overrides apply on top of any backend selection above.
if [[ -n "$ADJUDICATOR_MODEL" ]]; then
  LM_ARGS+=(--adjudicator-model "$ADJUDICATOR_MODEL")
fi
if [[ -n "$NARRATOR_MODEL" ]]; then
  LM_ARGS+=(--narrator-model "$NARRATOR_MODEL")
fi

OLLAMA_MODELS="/home/nisargparikh/Desktop/Fun Stuff/Mark_1/models" \
  python3 -m src.sim.autonomous \
    --world "worlds/${WORLD}" \
    --ticks "$TICKS" \
    "${LM_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    "${PASSTHRU[@]}"
