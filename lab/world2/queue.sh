#!/bin/bash
# Model-driven characters in the inn, one GPU job at a time (shares the lock
# with lab/methods/run_all.sh), each under the heat guard.
cd "$(dirname "$0")/../.."
job() {   # job <policy> <args...>
  (
    flock 9
    echo "=== $1 $(date +%H:%M)"
    PYTHONPATH=. lab/.venv/bin/python -m lab.world2.run "$@" 2>&1 | grep -vE "Warn|warn|Fetching|Loading weights|it/s\]" &
    sleep 5; lab/tools/heat_guard.sh "[l]ab.world2.run $1" &
    wait
  ) 9>lab/results/gpu.lock
}
job bias:gdecide:all --seeds 20
job text:gdecide:all --seeds 20
lab/.venv/bin/python -m lab.world2.run compare
echo "WORLD DONE $(date +%H:%M)"
