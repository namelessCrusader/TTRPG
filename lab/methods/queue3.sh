#!/bin/bash
# 2026-09-30: A7 (more personality scenes, two trait axes) and B5 (the inn with
# personality from appraisal), from docs/mind-lab-todo.md. Every GPU job takes
# the shared GPU lock (one model at a time, also across the other queues) and
# runs under the heat guard. Each job skips what it already finished.
#   nohup lab/methods/queue3.sh > lab/results/methods/queue3.log 2>&1 &
cd "$(dirname "$0")/../.."
PY=lab/.venv/bin/python

stage() {   # stage <experiment>: one stages.py job
  (
    flock 9
    echo "=== stages $* $(date +%H:%M)"
    PYTHONPATH=. $PY -m lab.methods.stages run "$@" 2>&1 | grep -vE "Warn|warn|Fetching|Loading weights|it/s\]" &
    sleep 5; lab/tools/heat_guard.sh "[l]ab.methods.stages run $1" &
    wait
  ) 9>lab/results/gpu.lock
}

job() {     # job <policy> <args...>: one night-by-night inn run, in the fixed world (world2b)
  (
    flock 9
    echo "=== inn $1 $(date +%H:%M)"
    LAB_WORLD_OUT=world2b PYTHONPATH=. $PY -m lab.world2.run "$@" 2>&1 | grep -vE "Warn|warn|Fetching|Loading weights|it/s\]" &
    sleep 5; lab/tools/heat_guard.sh "[l]ab.world2.run $1" &
    wait
  ) 9>lab/results/gpu.lock
}

# A7. the personality lists on two axes (temper, kindness), 28 scenes
stage personality2

# B5. the inn after the take/drop fix: the no-model policies (CPU), then the models
LAB_WORLD_OUT=world2b $PY -m lab.world2.run rules --seeds 10
LAB_WORLD_OUT=world2b $PY -m lab.world2.run random --seeds 10
job appraise:gdecide:all --seeds 10
job bias:gdecide:all --seeds 10

# reports (CPU)
LAB_WORLD_OUT=world2b $PY -m lab.world2.run compare
$PY -m lab.methods.stages report
echo "QUEUE3 DONE $(date +%H:%M)"
