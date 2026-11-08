#!/bin/bash
# Chunk 2 of the MODEL lane: fp16 x3 and bf16 x1 (the approximation's memory
# and speed), batches of 8 and 16 (throughput), and the exact load under a
# 1200 MB cap (its page cache is reclaimable). Under the test lock (the
# caller) and the GPU lock with the heat guard (job.sh).
cd "$(dirname "$0")/../../.."
Q=lab/results/m9/q
R=lab/results/opt/runs
J="lab/results/opt/job.sh load_bench run"
A="--req $Q/rooms.jsonl --req $Q/voxel.jsonl"
export JOB_TIMEOUT=170
for i in 1 2 3; do
  rm -rf $R/fp16_$i; LAB_GLINER_DTYPE=fp16 $J $R/fp16_$i $A --n 500
done
rm -rf $R/bf16_1; LAB_GLINER_DTYPE=bf16 $J $R/bf16_1 $A --n 500
for b in 8 16; do rm -rf $R/batch${b}_1; $J $R/batch${b}_1 $A --n 500 --batch $b; done
rm -rf $R/cap1200; MEM_CAP=1200M /home/nisargparikh/.cache/mark1-build/cap.sh $J $R/cap1200 $A --n 100
