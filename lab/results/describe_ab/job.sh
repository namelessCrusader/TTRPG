#!/bin/bash
# One GPU job of wave 10's LAB10 lane, under the GPU lock and the heat guard
# (as lab/world2/queue.sh and lab/results/opt/job.sh). The caller holds the
# test lock and the memory cap:
#   flock WORK/test.lock WORK/cap.sh lab/results/describe_ab/job.sh <module> <args...>
# Every job ends within JOB_TIMEOUT seconds (default 590: chunks of <= 10 min).
cd "$(dirname "$0")/../../.."
mod="$1"; shift
(
  flock 9
  echo "=== $mod $* $(date +%T) load=${LAB_GLINER_LOAD:-direct} dtype=${LAB_GLINER_DTYPE:-default}"
  PYTHONPATH=.${JOB_PATH:+:$JOB_PATH} timeout "${JOB_TIMEOUT:-590}" lab/.venv/bin/python -m "lab.methods.$mod" "$@" 2>&1 \
    | grep -vE "Warn|warn|Fetching|Loading weights|it/s\]|instantiate a model|self.encoder|====|Encoder model|Counting layer|Token pooling|Model Configuration|^$" &
  sleep 3; # THE PATTERN NAMES THIS JOB ONLY: the python of the lab venv running the
  # module, dots literal. "[l]ab.methods.$mod" also matched any command line
  # naming lab/methods/<mod>.py (a stepcheck), so the guard never ended and
  # held both locks (wave 11a, about 11 min).
  lab/tools/heat_guard.sh "[l]ab/\.venv/bin/python -m lab\.methods\.$mod( |$)" &
  wait
  echo "=== end $(date +%T)"
) 9>lab/results/gpu.lock
