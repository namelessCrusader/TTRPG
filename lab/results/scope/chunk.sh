#!/bin/bash
# One GPU chunk of the scope questions (lab/methods/scope_eval.py), under the
# test lock, the memory cap, the GPU lock and the heat guard (describe_ab/job.sh):
#   lab/results/scope/chunk.sh N      (the log goes to lab/results/scope/logs/chunkN.log)
cd "$(dirname "$0")/../../.."
S=lab/results/scope
flock "$HOME/.cache/mark1-build/test.lock" "$HOME/.cache/mark1-build/cap.sh" lab/results/describe_ab/job.sh \
  score_requests gdecide:all $S/q/todo.jsonl $S/d/all.jsonl --budget 540 --mem $S/logs/mem_$1.json \
  > $S/logs/chunk$1.log 2>&1
