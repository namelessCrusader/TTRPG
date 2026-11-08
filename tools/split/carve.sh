#!/usr/bin/env bash
# THE WHOLE SPLIT, AS ONE CHANGE.   carve.sh <repo> <work dir>
#
# Preconditions (refuses otherwise):
#   * the files the carve reads and writes are the BASE: src/voxel, src/mpm and
#     the 5 voxel test files of <repo> are equal to those in $BASE_DIR, the
#     base snapshot (a directory, used in place of a commit). Other paths may
#     differ, so the mind lane can go on while the carve runs.
#   * $BASE_DIR's src/voxel/sim.py is the exact file the map was derived from
#   * nobody edits src/voxel/sim.py until the gate passes (a freeze of ~20 min)
#
# What it does:
#   1. records the baseline FROM THE BASE: names, golden --all. It reuses
#      $BASE_GOLDEN when `golden.py fresh` says that recording was made from
#      these files by this golden.py.
#   2. carves a pristine copy into <work>/carved and proves it (astsame)
#   3. re-checks the repo against the base, then copies the carved modules in,
#      makes the one test edit (by content), drops sim's stale numba caches,
#      and writes the old->new location map
# Then run gate.sh. Bisect a failing gate with `split.py --only <mixins>`.
#
# Environment (required): LOCK (the shared lock file), BASE_DIR (the base
# snapshot directory). Optional: BASE_GOLDEN (default ${BASE_DIR}_golden.json).
# CARVE_SKIP_BASE=1 skips the comparison with the base (a run on a sandbox
# copy; BASE_DIR may then be unset); CARVE_SKIP_GOLDEN=1 skips the golden
# baseline.
set -euo pipefail
R=$(readlink -f "$1"); W=$(readlink -f "$2"); D=$(dirname "$(readlink -f "$0")")
die() { echo "carve.sh: $*" >&2; exit 1; }
[ -n "${LOCK:-}" ] || die "set LOCK to the shared lock file"
[ -n "${BASE_DIR:-}" ] || [ -n "${CARVE_SKIP_BASE:-}" ] \
  || die "set BASE_DIR to the base snapshot directory"
BASE_DIR=${BASE_DIR:+$(readlink -f "$BASE_DIR")}
BASE_GOLDEN=${BASE_GOLDEN:-${BASE_DIR:+${BASE_DIR}_golden.json}}
GOLDEN=$(readlink -f "$D/../golden.py")
EXPECT=4d0b87e91545fd2d32df2ad2f05985549b39d5f31c921029c64e3888ae1cf496
# What the carve reads (sim.py, the package it is copied with) and writes.
CARVED="src/voxel src/mpm tests/test_voxel.py tests/test_wounds.py
        tests/test_sound.py tests/test_projectiles.py tests/test_mpm.py"
sha() { sha256sum "$1" | cut -d' ' -f1; }
# Every carved path of the repo that is not equal to the base (empty: all equal).
not_base() {
  for p in $CARVED; do
    diff -rq -x __pycache__ "$BASE_DIR/$p" "$R/$p" 2>&1 || true
  done
}

# ── 0. preconditions ────────────────────────────────────────────────────────
if [ -z "${CARVE_SKIP_BASE:-}" ]; then
  [ -d "$BASE_DIR/src/voxel" ] || die "no base snapshot at $BASE_DIR (set BASE_DIR)"
  [ "$(sha "$BASE_DIR/src/voxel/sim.py")" = "$EXPECT" ] \
    || die "the base's sim.py is not the file the map was made from: re-run deps.py, splitmap.py, re-derive REHOME, re-pin"
  DIFF=$(not_base)
  [ -z "$DIFF" ] || die "the repo is not the base $BASE_DIR in what the carve touches; make a new snapshot first:
$DIFF"
fi
[ "$(sha "$R/src/voxel/sim.py")" = "$EXPECT" ] || die "working sim.py differs from the pinned input"
[ ! -e "$R/src/voxel/will.py" ] || die "src/voxel/will.py exists: already carved?"
mkdir -p "$W"

# ── 1. baseline, from the base ──────────────────────────────────────────────
P=$W/pristine
rm -rf "$P"; mkdir -p "$P/src"
cp "$R/src/__init__.py" "$P/src/"; cp -r "$R/src/voxel" "$R/src/mpm" "$P/src/"
find "$P" -name __pycache__ -prune -exec rm -rf {} +
python3 "$D/reexport.py" dump "$P" > "$W/names_before.json"
if [ -z "${CARVE_SKIP_GOLDEN:-}" ]; then
  rm -f "$W/base.json"
  if [ -f "$BASE_GOLDEN" ] && MARK1_REPO="$P" flock "$LOCK" python3 "$GOLDEN" fresh "$BASE_GOLDEN"; then
    cp "$BASE_GOLDEN" "$W/base.json"
  else
    MARK1_REPO="$P" flock "$LOCK" python3 "$GOLDEN" record "$W/base.json" --all
  fi
fi

# ── 2. carve and prove, outside the repo ────────────────────────────────────
rm -rf "$W/carved"
python3 "$D/split.py" "$P/src/voxel/sim.py" "$W/carved"
python3 "$D/astsame.py" "$P/src/voxel/sim.py" "$W/carved/src/voxel"

# ── 3. land ─────────────────────────────────────────────────────────────────
[ "$(sha "$R/src/voxel/sim.py")" = "$EXPECT" ] || die "sim.py changed during the carve: nothing copied"
if [ -z "${CARVE_SKIP_BASE:-}" ]; then
  DIFF=$(not_base)
  [ -z "$DIFF" ] || die "the repo changed during the carve: nothing copied
$DIFF"
fi
for m in sim kernels materials policy fields fluids heat air support bodies \
         skeleton pose persons physiology senses motion hands will ledgers; do
  cp "$W/carved/src/voxel/$m.py" "$R/src/voxel/$m.py"
done
# The one test edit: the test that rebinds _SWEEP_RAD must patch the module
# that reads it. By CONTENT, inside that one test: test_voxel.py has a second,
# identical `import src.voxel.sim as S` (the wind-up test) that must not move.
python3 - "$R/tests/test_voxel.py" <<'EOF'
import sys
p = sys.argv[1]; s = open(p).read()
i = s.index("def test_an_ARM_CANNOT_SWEEP_THROUGH_what_it_would_CLEAR_at_the_end")
j = s.index("\ndef test_", i + 10)
body = s[i:j]
assert body.count("import src.voxel.sim as S") == 1, "test changed shape"
open(p, "w").write(s[:i] + body.replace("import src.voxel.sim as S",
                                        "import src.voxel.pose as S") + s[j:])
EOF
# numba keys its cache by source file: sim.*.nbi/.nbc are orphans from now on
rm -f "$R"/src/voxel/__pycache__/sim.*.nbi "$R"/src/voxel/__pycache__/sim.*.nbc
# where every old sim.py:N went (docs and notes cite them)
if [ -d "$R/docs" ]; then
  python3 "$D/wheremap.py" "$P/src/voxel/sim.py" "$R/src/voxel" > "$R/docs/sim-split-map.md"
fi
echo "carved: $(wc -l < "$R/src/voxel/sim.py") lines left in sim.py; now: gate.sh $R $W"
