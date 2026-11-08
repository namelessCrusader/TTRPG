#!/usr/bin/env bash
# THE GATE for the carve (and for any later behaviour-preserving refactor of
# src/voxel).   gate.sh <repo> <work dir from carve.sh>
# Every step must exit 0; `set -e` makes a pytest exit 5 (nothing collected,
# e.g. a node id deselected by -m) a failure like any other. No -m anywhere.
# Environment: LOCK (the shared lock file, required); GATE_SKIP_GOLDEN,
# GATE_SKIP_TESTS and GATE_SKIP_HARVEST skip those parts (for a trial run only).
set -euo pipefail
R=$(readlink -f "$1"); W=$(readlink -f "$2"); D=$(dirname "$(readlink -f "$0")")
[ -n "${LOCK:-}" ] || { echo "gate.sh: set LOCK to the shared lock file" >&2; exit 1; }
GOLDEN=$(readlink -f "$D/../golden.py")
cd "$R"
python3 -c "import src.voxel.sim"                                        # imports, no cycle
python3 "$D/astsame.py" "$W/pristine/src/voxel/sim.py" "$R/src/voxel"     # IDENTICAL
MARK1_REPO="$R" python3 "$D/namecheck.py"                                # clean
python3 "$D/reexport.py" dump "$R" > "$W/names_now.json"
python3 "$D/reexport.py" diff "$W/names_before.json" "$W/names_now.json"  # 0 missing, 0 changed
python3 "$D/reexport.py" shared "$R"                                      # one object per table
[ -n "${GATE_SKIP_GOLDEN:-}" ] || \
  MARK1_REPO="$R" flock "$LOCK" python3 "$GOLDEN" check "$W/base.json" --all   # ALL SAME
[ -n "${GATE_SKIP_TESTS:-}" ] || \
  flock "$LOCK" python3 -m pytest -p no:cacheprovider -q --durations=0 \
    tests/test_voxel.py tests/test_wounds.py tests/test_sound.py \
    tests/test_projectiles.py tests/test_mpm.py tests/test_module_hygiene.py \
    > "$W/durations.txt"
[ -n "${GATE_SKIP_HARVEST:-}" ] || \
  flock "$LOCK" python3 - "$W" <<'EOF'
import sys
sys.path.insert(0, ".")
from src.voxel import harvest
got = harvest.to_file(sys.argv[1] + "/harvest32.jsonl", range(32), workers=1)
print(got)
assert got["broke"] == 0, "a world fell over: harvest turns exceptions into rows"
EOF
# A trial run that skipped a part has not passed the gate: say so.
SKIPPED="${GATE_SKIP_GOLDEN:+golden }${GATE_SKIP_TESTS:+tests }${GATE_SKIP_HARVEST:+harvest}"
if [ -n "$SKIPPED" ]; then
  echo "GATE PASSED IN PART ONLY: skipped ${SKIPPED% }"
else
  echo "GATE PASSED"
fi
