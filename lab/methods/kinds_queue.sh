#!/bin/bash
# The "people" and "places" kinds (plan §4), every factor, one model at a time.
# Shares the GPU lock with the other queues. Fast models first.
cd "$(dirname "$0")/../.."
Q=lab/methods/run_all.sh
for kind in people places; do
  $Q --cases lab/methods/cases_$kind.jsonl --tag $kind -- gdecide:all bev:all von13:all onli08b:one \
      qwen17:one_cal laya_en:all bge:apart qwen06:one
done
for kind in people places; do
  $Q --cases lab/methods/cases_$kind.jsonl --tag $kind -- qwen17:two laya_en:two
  LAB_GPU_FACTOR=1.1 $Q --cases lab/methods/cases_$kind.jsonl --tag $kind -- onli2b:one
done
echo "KINDS DONE $(date +%H:%M)"
