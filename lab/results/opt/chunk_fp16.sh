#!/bin/bash
# The fp16 approximation over every M9 request (lab/results/m9/q), into
# lab/results/opt/m9_fp16/d, in chunks: each run stops after BUDGET seconds and
# the next one continues where it stopped. Under the test lock (the caller)
# and the GPU lock with the heat guard (job.sh).
cd "$(dirname "$0")/../../.."
Q=lab/results/m9/q
D=lab/results/opt/m9_fp16/d
n=$(ls lab/results/opt/m9_fp16/mem_*.json 2>/dev/null | wc -l)
LAB_GLINER_DTYPE=${DTYPE:-fp16} JOB_TIMEOUT=590 lab/results/opt/job.sh score_requests gdecide:all \
  $Q/rooms.jsonl $D/rooms.jsonl --also $Q/voxel.jsonl $D/voxel.jsonl \
  --also $Q/rooms_lab.jsonl $D/rooms_lab.jsonl --also $Q/rooms_full.jsonl $D/rooms_full.jsonl \
  --also $Q/voxel_perm.jsonl $D/voxel_perm.jsonl --also $Q/rooms_perm.jsonl $D/rooms_perm.jsonl \
  --budget ${BUDGET:-500} --mem lab/results/opt/m9_fp16/mem_$((n + 1)).json
