#!/bin/bash
# pause a lab job while the CPU runs hot, resume when it cools. $1 = pgrep pattern
S="$(cd "$(dirname "$0")/../results" && pwd)"
pat="$1"
sleep 3
while pgrep -f "$pat" >/dev/null; do
  t=$(sensors | awk '/Tctl/ {gsub(/[+°C]/,"",$2); print int($2)}')
  pids=$(pgrep -f "$pat")
  if [ "$t" -ge 88 ]; then kill -STOP $pids; echo "$(date +%T) $t C paused" >> "$S/watchdog.log"; fi
  if [ "$t" -le 80 ]; then kill -CONT $pids 2>/dev/null; fi
  sleep 2
done
