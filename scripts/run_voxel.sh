#!/usr/bin/env bash
# Deprecated wrapper — use ./run_sim.sh --pygame voxel_tavern
exec "$(dirname "$0")/run_sim.sh" --pygame "${1:-voxel_tavern}" "${@:2}"
