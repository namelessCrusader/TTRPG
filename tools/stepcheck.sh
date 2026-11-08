#!/usr/bin/env bash
# ONE STEP'S GOLDEN CHECK, IN ITS OWN SANDBOX (docs/sim-mind-design.md, 4.1).
#
#     tools/stepcheck.sh <id> "<declared cases | none | any>" <owned paths...>
#
# Several agents edit the live tree at the same time, so a check must never
# see the half-done work of another agent. This script checks a tree that has
# only the base and the step's own files:
#   1. Once, at the step's start: it copies the base snapshot $BASE_DIR to
#      $WORK/<id>/head, keeps a copy of tools/golden.py as $WORK/<id>/golden.py,
#      and gets a golden base for head. It uses $BASE_GOLDEN when
#      `golden.py fresh` says that recording was made from the same files by
#      the same golden.py. Otherwise it records a new one in head.
#   2. At each check: it makes $WORK/<id>/mine from head plus the step's owned
#      paths, copied from the live tree. An owned path that is not in the live
#      tree is removed from mine (the step deleted it).
#   3. It runs `golden.py check --all --allow <declared>` in mine. The check
#      passes when every case that is not declared is SAME. NEW names are
#      listed and do not fail the check.
#
# An owned path is relative to the repo root. A bare file name that is not at
# the root means src/voxel/<name>, as the design writes it ("ledgers.py").
# "." means the whole snapshot (src tests tools lab/world2 lab/judge, without
# tools/blender): use it for the wave-end check.
#
# Environment (WORK, LOCK and BASE_DIR are required; the orchestrator sets them):
#   WORK         the scratch root; the step works in $WORK/<id>
#   LOCK         the shared lock file; every python run takes it
#   BASE_DIR     the base snapshot: a directory with src tests tools lab/world2
#                lab/judge, used in place of a commit
#   BASE_GOLDEN  a golden recording of BASE_DIR (default: ${BASE_DIR}_golden.json)
#   ONLY         a comma list of cases to check in place of --all, for quick
#                loops; the step's final check must run without it
#   TESTS        pytest files or node ids to run inside mine after the golden
#                check (never with -m: pytest exits 5 and that is a failure)
#   LAWS         "default" or a comma list of scenes: after the tests, print the
#                per-law mean and p99 ms per tick of mine (tools/law_timing.py,
#                which mine must hold: list it among the owned paths)
#
# To start a step again from a new base: rm -rf "$WORK/<id>".
# MARK1_CAP: a command that runs its arguments under a memory cap (the
# scratchpad cap.sh); every python run below goes through it. Unset: no cap.
CAP="${MARK1_CAP:-env}"
set -euo pipefail
if [ $# -lt 3 ]; then
  sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'
  exit 2
fi
ID=$1; DECLARED=$2; shift 2
R=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
die() { echo "stepcheck.sh: $*" >&2; exit 1; }
[ -n "${WORK:-}" ] || die "set WORK to the scratch root"
[ -n "${LOCK:-}" ] || die "set LOCK to the shared lock file"
[ -n "${BASE_DIR:-}" ] || die "set BASE_DIR to the base snapshot directory"
[ -d "$BASE_DIR/src" ] || die "no base snapshot at $BASE_DIR (set BASE_DIR)"
BASE_DIR=$(readlink -f "$BASE_DIR")
BASE_GOLDEN=${BASE_GOLDEN:-${BASE_DIR}_golden.json}
W=$WORK/$ID; HEAD=$W/head; MINE=$W/mine; G=$W/golden.py
# lab/methods holds 21 MB of case data; only its code is part of a tree
# (tests/test_mind_record.py reads run.py and runs score_requests.py)
SNAP="src tests tools lab/world2 lab/judge lab/__init__.py $(cd "$R" && ls lab/methods/*.py 2>/dev/null | tr '\n' ' ')"

# Copy one repo-relative path from tree $1 into tree $2, exactly. A path that
# $1 does not have is removed from $2. __pycache__ is never copied, and is
# kept where it is, so numba compiles once per sandbox and not once per check.
put() {
  local from=$1 to=$2 rel=$3
  if [ -d "$from/$rel" ]; then
    mkdir -p "$to/$rel"
    rsync -a --delete --exclude __pycache__ --exclude /blender "$from/$rel/" "$to/$rel/"
  elif [ -e "$from/$rel" ]; then
    mkdir -p "$(dirname "$to/$rel")"
    rsync -a "$from/$rel" "$to/$rel"
  else
    rm -rf "${to:?}/$rel"
  fi
}

# ── 1. once per step: head, the frozen golden.py, the golden base ───────────
mkdir -p "$W"
if [ -d "$HEAD" ]; then
  FROM=$(cat "$W/head.from" 2>/dev/null || echo "an unknown base")
  [ "$FROM" = "$BASE_DIR" ] \
    || die "$HEAD was copied from $FROM, not $BASE_DIR: rm -rf $W to start again"
else
  rm -rf "$HEAD.part"
  rsync -a --exclude __pycache__ "$BASE_DIR/" "$HEAD.part/"
  mv "$HEAD.part" "$HEAD"
  echo "$BASE_DIR" > "$W/head.from"
fi
[ -f "$G" ] || cp "$R/tools/golden.py" "$G"
cmp -s "$G" "$R/tools/golden.py" \
  || echo "stepcheck: NOTE: tools/golden.py changed after this step started; using the copy from its start"
if [ ! -f "$W/base.json" ]; then
  if [ -f "$BASE_GOLDEN" ] && MARK1_REPO="$HEAD" flock "$LOCK" $CAP python3 "$G" fresh "$BASE_GOLDEN"; then
    cp "$BASE_GOLDEN" "$W/base.json"
  else
    echo "stepcheck: recording the golden base in $HEAD (about 4 minutes)"
    MARK1_REPO="$HEAD" flock "$LOCK" $CAP python3 "$G" record "$W/base.json" --all
  fi
fi

# ── 2. mine = head + the owned paths from the live tree ─────────────────────
rsync -a --delete --exclude __pycache__ "$HEAD/" "$MINE/"
OWNED=()
for p in "$@"; do
  p=${p%/}
  if [ "$p" = "." ]; then
    OWNED+=($SNAP)
  elif [ -e "$R/$p" ] || [ -e "$HEAD/$p" ] || [[ $p == */* ]]; then
    OWNED+=("$p")
  else
    OWNED+=("src/voxel/$p")
  fi
done
for rel in "${OWNED[@]}"; do
  [ -e "$R/$rel" ] || [ -e "$HEAD/$rel" ] || echo "stepcheck: $rel is in neither tree yet"
  put "$R" "$MINE" "$rel"
done
echo "stepcheck $ID: mine differs from head in:"
diff -rq -x __pycache__ "$HEAD" "$MINE" | sed "s|$W/||g; s/^/  /" || true

# ── 3. the golden check, then the step's tests, inside mine ─────────────────
SCOPE=(--all); NOTE=""
if [ -n "${ONLY:-}" ]; then SCOPE=(--only "$ONLY"); NOTE=" ONLY $ONLY, not the full suite"; fi
set +e
MARK1_REPO="$MINE" flock "$LOCK" $CAP python3 "$G" check "$W/base.json" "${SCOPE[@]}" \
  --allow "$DECLARED" | tee "$W/check.log"
GRC=${PIPESTATUS[0]}
TRC=0
if [ -n "${TESTS:-}" ]; then
  # shellcheck disable=SC2086  # TESTS is a list of words on purpose
  (cd "$MINE" && flock "$LOCK" $CAP python3 -m pytest -q -p no:cacheprovider $TESTS) \
    | tee "$W/tests.log"
  TRC=${PIPESTATUS[0]}
fi
# ── 4. the per-law timing, if asked (LAWS=default, or a comma list of scenes) ─
# tools/law_timing.py of the lane's own tree: mean and p99 ms per tick of every law.
if [ -n "${LAWS:-}" ] && [ -f "$MINE/tools/law_timing.py" ]; then
  SCENES=${LAWS/default/torch_pillar,forge,pond}
  (cd "$MINE" && MARK1_REPO="$MINE" flock "$LOCK" $CAP python3 tools/law_timing.py \
      --scenes "$SCENES" --repeat 3 --json "$W/law_timing.json") | tee "$W/law_timing.log" || true
fi
set -e
if [ "$GRC" -eq 0 ] && [ "$TRC" -eq 0 ]; then
  echo "STEPCHECK $ID PASSED (declared: $DECLARED)$NOTE"
else
  echo "STEPCHECK $ID FAILED (golden exit $GRC, tests exit $TRC; declared: $DECLARED)$NOTE"
  exit 1
fi
