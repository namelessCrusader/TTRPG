#!/bin/bash
# One GPU job of wave O's MODEL lane, under the GPU lock and the heat guard
# (as lab/world2/queue.sh). The caller holds the test lock and the memory cap:
#   flock WORK/test.lock WORK/cap.sh lab/results/opt/job.sh <module> <args...>
# e.g. lab/results/opt/job.sh load_bench run lab/results/opt/runs/x --req ... --n 500
cd "$(dirname "$0")/../../.."
mod="$1"; shift
(
  flock 9
  echo "=== $mod $* $(date +%T) load=${LAB_GLINER_LOAD:-direct} dtype=${LAB_GLINER_DTYPE:-default}"
  PYTHONPATH=. timeout "${JOB_TIMEOUT:-590}" lab/.venv/bin/python -m "lab.methods.$mod" "$@" 2>&1 \
    | grep -vE "Warn|warn|Fetching|Loading weights|it/s\]|instantiate a model|self.encoder|====|Encoder model|Counting layer|Token pooling|Model Configuration|^$" &
  sleep 3; # THE PATTERN NAMES THIS JOB ONLY: the python of the lab venv running the
  # module, dots literal. "[l]ab.methods.$mod" also matched any command line
  # naming lab/methods/<mod>.py (a stepcheck), so the guard never ended and
  # held both locks (wave 11a, about 11 min).
  lab/tools/heat_guard.sh "[l]ab/\.venv/bin/python -m lab\.methods\.$mod( |$)" &
  wait
  echo "=== end $(date +%T)"
) 9>lab/results/gpu.lock
