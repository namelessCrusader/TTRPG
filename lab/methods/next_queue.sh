#!/bin/bash
# After plan v4 (2026-09-30): the short GPU jobs B2, B4, B1 from docs/mind-lab-todo.md.
# Every GPU job takes the shared GPU lock, so this waits for any other queue
# (the question battery, B6) and runs one model at a time, under the heat guard.
# Each job skips what it already finished, so a stopped queue can be started again.
#   nohup lab/methods/next_queue.sh > lab/results/methods/next_queue.log 2>&1 &
cd "$(dirname "$0")/../.."
Q=lab/methods/run_all.sh
PY=lab/.venv/bin/python

stage() {   # stage <experiment>: one stages.py job, under the lock and the heat guard
  (
    flock 9
    echo "=== stages $* $(date +%H:%M)"
    PYTHONPATH=. $PY -m lab.methods.stages run "$@" 2>&1 | grep -vE "Warn|warn|Fetching|Loading weights|it/s\]" &
    sleep 5; lab/tools/heat_guard.sh "[l]ab.methods.stages run $1" &
    wait
  ) 9>lab/results/gpu.lock
}

# B2. the mention lure: the state in three wordings, and the general lure
$Q --cases lab/methods/cases_state2.jsonl --tag state2 --keep-scores -- gdecide:all
$Q --cases lab/methods/cases_lure.jsonl --tag lure --keep-scores -- gdecide:all
# B4. relevance with the threshold A5 found
stage relevance2
# B1. the capability looked up from the verb
stage capability2

# reports (CPU)
$PY -m lab.methods.primitives report
$PY -m lab.methods.cases_state report2
$PY -m lab.methods.stages report
echo "NEXT DONE $(date +%H:%M)"
