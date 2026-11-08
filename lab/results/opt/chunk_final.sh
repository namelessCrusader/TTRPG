#!/bin/bash
# Chunk 3 of the MODEL lane: the new default (no switches) x3 on the same 500
# requests, and the switch back (LAB_GLINER_LOAD=old LAB_GLINER_DTYPE=fp32) x1.
cd "$(dirname "$0")/../../.."
Q=lab/results/m9/q
R=lab/results/opt/runs
J="lab/results/opt/job.sh load_bench run"
A="--req $Q/rooms.jsonl --req $Q/voxel.jsonl --n 500"
export JOB_TIMEOUT=170
for i in 1 2 3; do rm -rf $R/default_$i; $J $R/default_$i $A; done
rm -rf $R/switch_old; LAB_GLINER_LOAD=old LAB_GLINER_DTYPE=fp32 $J $R/switch_old $A
