import random, sys
sys.argv = ["x"]
exec(open("lab/listsize.py").read().split("a = laya.load")[0])   # NEEDS, JUNK
import laya
from laya.shortlist import predict_shortlist, embed_fn_from_agent
a = laya.load("convaiinnovations/laya")
embed = embed_fn_from_agent(a)
def run(N, mode):
    rng = random.Random(N)
    hits = 0
    for need, item in NEEDS:
        opts = rng.sample([j for j in JUNK if j != item], N - 1) + [item]
        rng.shuffle(opts)
        q = {"pick": {"type": "choice", "instructions": "What does Kael take?",
                      "criteria": {o: None for o in opts}}}
        st = f"Kael needs to {need}. He looks at what is in the storeroom."
        if mode == "shortlist":
            r = predict_shortlist(a, st, q, embed, k=16)
        else:
            r = a.predict(st, q)
        hits += r["answers"]["pick"]["choice"] == item
    return hits
for N in (16, 32, 64, 80):
    a.cfg["max_len"], a.cfg["head_max_len"] = 512, 192
    base = run(N, "plain")
    a.cfg["max_len"], a.cfg["head_max_len"] = 1536, 1400
    big = run(N, "plain")
    a.cfg["max_len"], a.cfg["head_max_len"] = 512, 192
    sl = run(N, "shortlist")
    print(f"N={N:3}  budget 192: {base}/20   budget 1400: {big}/20   embed-shortlist to 16: {sl}/20", flush=True)
