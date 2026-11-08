#!/bin/bash
# Creativity tests, layers 1 and 2: every scorer over cases_creative.jsonl,
# one at a time, under the heat guard. Each runner skips what it finished.
cd "$(dirname "$0")/../.."
Q=lab/methods/run_all.sh
A=(--cases lab/methods/cases_creative.jsonl --tag creative)
$Q "${A[@]}" -- gdecide:all bev:all von13:all laya_en:all laya_en:solo bge:apart \
    onli08b:one qwen06:one qwen06:one_cal qwen17:one qwen17:one_cal laya_en:two qwen17:two
LAB_GPU_FACTOR=1.1 $Q "${A[@]}" -- onli2b:one
lab/.venv/bin/python -m lab.methods.creative report
echo "CREATIVE DONE $(date +%H:%M)"
