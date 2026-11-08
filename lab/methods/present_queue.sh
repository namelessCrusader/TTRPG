#!/bin/bash
# The presentation battery (lab/methods/present.py). Shares the GPU lock.
cd "$(dirname "$0")/../.."
Q=lab/methods/run_all.sh
A=(--cases lab/methods/cases_present.jsonl --tag present)
$Q "${A[@]}" -- overlap:all gdecide:all bev:all von13:all laya_en:all onli08b:one qwen17:one_cal
# how query and options combine: the default input only
$Q "${A[@]}" --only present=default -- qwen17:cloze qwen17:cloze_cal gdecide:hier bev:hier laya_en:hier
LAB_GPU_FACTOR=1.1 $Q "${A[@]}" -- onli2b:one
LAB_GPU_FACTOR=1.1 $Q "${A[@]}" --only present=default -- onli2b:use onli2b:work
$Q "${A[@]}" -- qwen17:two
lab/.venv/bin/python -m lab.methods.present report
echo "PRESENT DONE $(date +%H:%M)"
