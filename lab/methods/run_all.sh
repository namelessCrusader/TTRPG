#!/bin/bash
# Run scorers ONE AT A TIME, each under the heat guard. Usage: run_all.sh [runner args...] -- scorer...
cd "$(dirname "$0")/../.."
args=(); while [ "$1" != "--" ] && [ -n "$1" ]; do args+=("$1"); shift; done; shift
for s in "$@"; do
  echo "=== $s $(date +%H:%M)"
  py=lab/.venv/bin/python
  case "$s" in von13:*) py=lab/vendor/.venv-von13/bin/python;; esac   # Von 1.3 has its own environment
  # ONE GPU JOB AT A TIME, even across queues started separately: the lock
  # is held for the whole scorer, so another queue waits its turn.
  (
    flock 9
    PYTHONPATH=. $py -m lab.methods.run "$s" "${args[@]}" 2>&1 | grep -vE "Warn|warn|return Agent|Fetching|Loading weights|it/s\]" &
    sleep 5; lab/tools/heat_guard.sh "[l]ab.methods.run $s" &
    wait
  ) 9>lab/results/gpu.lock
done
echo "ALL DONE $(date +%H:%M)"
