#!/bin/bash
# The re-runs paused on 2026-09-28: GLiNER2.5-Decide after its device fix,
# openjev 2B with a looser memory check, and GLiNER-Decide's invariance test.
# Each runner skips what it already finished.
cd "$(dirname "$0")/../.."
Q=lab/methods/run_all.sh
$Q -- gdecide:all gdecide:tour4 gdecide:perm4 gdecide1b:all
LAB_GPU_FACTOR=1.1 $Q -- onli2b:one
$Q --only size=16,size=64 --perms 4 -- gdecide:all
echo "RERUN DONE $(date +%H:%M)"
