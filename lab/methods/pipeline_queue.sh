#!/bin/bash
# Short list, then choose (lab/methods/pipeline.py), both stages.
cd "$(dirname "$0")/../.."
Q=lab/methods/run_all.sh
$Q --only size=32,size=64,size=128 --keep-scores --tag scores -- bev:all gdecide:all onli08b:one
lab/.venv/bin/python -m lab.methods.pipeline build
$Q --cases lab/methods/cases_pipe.jsonl --tag pipe -- qwen17:two
LAB_GPU_FACTOR=1.1 $Q --cases lab/methods/cases_pipe.jsonl --tag pipe -- onli2b:one
lab/.venv/bin/python -m lab.methods.pipeline report
echo "PIPELINE DONE $(date +%H:%M)"
