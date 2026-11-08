#!/bin/bash
# The rest of the Workstream A queue, paused on 2026-09-28 for other work.
# Every runner skips cases it already finished, so this simply continues.
# One GPU job at a time; each under the heat guard (see run_all.sh).
cd "$(dirname "$0")/../.."
Q=lab/methods/run_all.sh
# 1. the main set: what was left of it
$Q -- qwen17:all qwen06:two qwen17:two von:one
# 2. the tournament (the user's idea), on list size and similarity
$Q --only size,similar -- laya_en:tour4 laya_en:tour8k2 laya_en:tour4x3 von:tour4
# 3. Laya one at a time, not as yes/no
$Q -- laya_en:solo laya_en:anchor
# 4. every scorer on the hand-picked near misses
$Q --only nearmiss -- laya_en:all laya_en:two laya_en:solo laya_en:anchor laya_en:one laya_multi:all von:all bge:apart qwen06:one qwen06:one_cal qwen17:one qwen17:one_cal qwen17:all bge_qwen17:short laya_en:tour4
# 5. GLiNER2.5-Decide (340M; its tournament; the 1B version)
$Q -- gdecide:all gdecide:tour4 gdecide1b:all
# 6. models that claim order invariance, and "ask in several orders and average"
#    bev-decider-0.4B (invariant by construction; CC-BY-NC, testing only),
#    Von 1.3 (invariant by construction; its own environment), openjev NLI
#    (each option judged alone), and Laya / GLiNER-Decide averaged over 4 orders
$Q -- bev:all von13:all onli08b:one onli2b:one laya_en:perm4 gdecide:perm4 bev:tour4
# 7. THE INVARIANCE TEST: the same case in 4 orders; does the pick change?
#    Checks the invariance claims, and measures the order bias of the rest.
$Q --only size=16,size=64 --perms 4 -- laya_en:all laya_en:perm4 von:all von13:all bev:all gdecide:all qwen06:all qwen17:one_cal onli08b:one
echo "QUEUE DONE $(date +%H:%M)"
