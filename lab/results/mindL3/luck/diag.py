import json,sys,numpy as np
from pathlib import Path
from lab.world2.lifetime import night_seed
from lab.world2.world import World
v=float(sys.argv[2])
for ch in ("hermit","rules","always-giving","tit-for-tat"):
    E=[];Z=[]
    for s in range(1,11):
        r=json.load(open(Path(sys.argv[1])/f"A_luck_{ch}_seed{s}_n100.json"))
        for n in range(20,100):
            c=World(seed=night_seed(s,n),flags="all4",luck=v).catch.get(r["focal"],None)
            if c is None: continue
            (Z if c==0 else E).append(r["err"][n])
    print(f"{ch:14s} focal error on nights with a catch {np.mean(E):.3f} (n={len(E)}), empty-catch nights {np.mean(Z):.3f} (n={len(Z)})")
