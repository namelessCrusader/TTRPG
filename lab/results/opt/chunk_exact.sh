#!/bin/bash
# Chunk 1 of the MODEL lane: the exact load against the old one, 3 runs each,
# taken in turn, 500 recorded requests (250 inn, 250 voxel). Under the test
# lock (the caller) and the GPU lock with the heat guard (job.sh).
cd "$(dirname "$0")/../../.."
Q=lab/results/m9/q
R=lab/results/opt/runs
for i in 1 2 3; do
  for how in direct old; do
    rm -rf $R/${how}_$i
    LAB_GLINER_LOAD=$how JOB_TIMEOUT=170 lab/results/opt/job.sh load_bench run $R/${how}_$i \
      --req $Q/rooms.jsonl --req $Q/voxel.jsonl --n 500
  done
done
