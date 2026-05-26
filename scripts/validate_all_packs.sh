#!/usr/bin/env bash
# Validate all shipped world packs (reactive smoke, no LM).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 -m src.sim.validate_pack --all "${1:-10}"
