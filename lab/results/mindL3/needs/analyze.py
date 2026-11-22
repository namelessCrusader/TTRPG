"""Pass rules for one run dir of the needs study. World X is relabelled to the name the 0a report code expects:
all5 -> all2, all5R0 -> noR (so 'extractor wins only where reprisal is weak' compares all5 with all5R0).
usage: analyze.py DIR WORLD [WORLD_R0]"""
import sys, json
import lab.mindL.tournament_report as tr
D, W = sys.argv[1], sys.argv[2]
W0 = sys.argv[3] if len(sys.argv) > 3 else None
R = tr.load(D)
R = {(k, {W: "all2", W0: "noR"}.get(w, w), c, s, rt): r for (k, w, c, s, rt), r in R.items()}
T = tr.tournament(R)
P = tr.pass_rules_a(T, R)
O = tr.oracle(R) if any(k[0] == "O" for k in R) else {"rows": {}}
B = tr.bcqw(R)
S = tr.stance_check(R)
print(tr.text(T, P, O if O["rows"] else {"rows": {}}, B, S))
print("\nBY VERB (all2 = the world under study), selected")
for v in ("serve", "give", "praise", "gossip", "strike", "insult", "accuse", "buy", "work", "tend", "bandage"):
    x = B["worlds"]["all2"]["by_verb"].get(v)
    if x: print(v, {k: (round(y, 3) if isinstance(y, float) else y) for k, y in x.items()})
