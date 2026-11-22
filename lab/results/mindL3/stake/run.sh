#!/bin/bash
cd "/home/nisargparikh/Desktop/Fun Stuff/Mark_1"
export PYTHONPATH="$PWD"
L=/home/nisargparikh/.cache/mark1-build/test.lock
C=/home/nisargparikh/.cache/mark1-build/cap.sh
stage() { # dir, args...
  d=$1; shift; mkdir -p "/home/nisargparikh/Desktop/Fun Stuff/Mark_1/lab/results/mindL3/stake/$d"
  echo "== $d $* $(date +%T)"
  MINDL3_RUNS="/home/nisargparikh/Desktop/Fun Stuff/Mark_1/lab/results/mindL3/stake/$d" flock $L $C nice -n 10 lab/.venv/bin/python -m lab.mindL.tournament run "$@" --workers 3
}
stage runs_all6 A --worlds all6
stage runs_all6 O --worlds all6
stage runs_all6 E --worlds all6
stage runs_all5 E --worlds all5
stage runs_no_lumpy A --worlds no_lumpy
stage runs_no_stock A --worlds no_stock
stage runs_no_choosy A --worlds no_choosy
stage runs_all6R0 A --worlds all6R0 --choosers rules,extractor
stage runs_all5 A --worlds all5
echo DONE $(date +%T)
