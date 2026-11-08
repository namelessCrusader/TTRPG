#!/bin/bash
# What does each model use? (lab/methods/ablate.py). Shares the GPU lock.
cd "$(dirname "$0")/../.."
Q=lab/methods/run_all.sh
A=(--cases lab/methods/cases_ablate.jsonl --tag ablate --keep-scores)
$Q "${A[@]}" -- gdecide:all bev:all von13:all laya_en:all qwen17:one_cal onli08b:one qwen17:two
LAB_GPU_FACTOR=1.1 $Q "${A[@]}" -- onli2b:one
lab/.venv/bin/python -m lab.methods.ablate report
echo "ABLATE DONE $(date +%H:%M)"
