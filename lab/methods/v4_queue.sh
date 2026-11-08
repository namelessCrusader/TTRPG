#!/bin/bash
# Plan v4 (2026-09-29): GLiNER-Decide first. WRITTEN, NOT STARTED. Start it with
#   nohup lab/methods/v4_queue.sh > lab/results/methods/v4_queue.log 2>&1 &
# Every GPU job takes the shared GPU lock (one model at a time, also across the
# older queues) and runs under the heat guard. Each job skips what it already
# finished, so a stopped queue can simply be started again.
cd "$(dirname "$0")/../.."
Q=lab/methods/run_all.sh
PY=lab/.venv/bin/python

stage() {   # stage <experiment> [args]: one stages.py job, under the lock and the heat guard
  (
    flock 9
    echo "=== stages $* $(date +%H:%M)"
    PYTHONPATH=. $PY -m lab.methods.stages run "$@" 2>&1 | grep -vE "Warn|warn|Fetching|Loading weights|it/s\]" &
    sleep 5; lab/tools/heat_guard.sh "[l]ab.methods.stages run $1" &
    wait
  ) 9>lab/results/gpu.lock
}

# 0. the ablation's "no need" cells again (its sentence was fixed)
$Q --cases lab/methods/cases_ablate.jsonl --tag ablate --keep-scores -- gdecide:all

# 1. capture: every option's vector and score, on every task space
$Q --tag cap_things --capture -- gdecide:all
$Q --cases lab/methods/cases_creative.jsonl --tag cap_creative --capture -- gdecide:all
$Q --cases lab/methods/cases_people.jsonl --tag cap_people --capture -- gdecide:all
$Q --cases lab/methods/cases_places.jsonl --tag cap_places --capture -- gdecide:all

# 3. the fair-die probe (a few seconds for each small model)
$Q --cases lab/methods/cases_die.jsonl --tag die --keep-scores -- gdecide:all bev:all von13:all laya_en:all \
    onli08b:one qwen17:one_cal

# 2. small questions first (GLiNER-Decide), with capture for the group-head test
stage capability --capture
stage people --capture
stage personality --capture
stage relevance --capture
stage state --capture
stage softkind
stage speech

# 4. scene state: the plain choice, with and without the state written out
$Q --cases lab/methods/cases_state.jsonl --tag state --keep-scores -- gdecide:all

# the word-overlap baseline on the same stages (CPU, no model)
for e in capability people personality relevance state softkind; do
  LAB_DEVICE=cpu LAB_DUTY=1 $PY -m lab.methods.stages run $e --scorer overlap:all
done

# reports (CPU)
$PY -m lab.methods.ablate report
$PY -m lab.methods.stages report
$PY -m lab.methods.cases_state report
$PY -m lab.methods.prior report
$PY -m lab.methods.learnable report
$PY -m lab.methods.examples
echo "V4 DONE $(date +%H:%M)"
