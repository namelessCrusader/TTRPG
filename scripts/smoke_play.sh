#!/usr/bin/env bash
# Quick automated check (no window): pygame client smoke tests
set -euo pipefail
cd "$(dirname "$0")/.."
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
python3 -m pytest \
  tests/test_play_client_smoke.py \
  tests/test_play_client_interactions.py \
  tests/test_player_spawn.py \
  tests/test_voxel_draw.py \
  -q
echo "smoke OK"
