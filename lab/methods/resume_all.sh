#!/bin/bash
# Restart every paused queue (2026-09-28 pause). Each runner skips the cases
# and nights it already finished; the queues share one GPU lock, so only one
# model is on the GPU at a time, each under the heat guard.
cd "$(dirname "$0")/../.."
R=lab/results
nohup lab/methods/ablate_queue.sh  >> $R/methods/ablate_queue.log  2>&1 &
nohup lab/world2/queue.sh          >> $R/world2_queue.log          2>&1 &
nohup lab/methods/present_queue.sh >> $R/methods/present_queue.log 2>&1 &
nohup lab/methods/kinds_queue.sh   >> $R/methods/kinds_queue.log   2>&1 &
echo "restarted 4 queues $(date +%H:%M)"
