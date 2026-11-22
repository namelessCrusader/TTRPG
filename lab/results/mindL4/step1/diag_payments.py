import collections, json
import lab.world2.partners as P
from lab.mindL import tournament as T
log=[]
orig=P.Need.answer
def ans(self,w,p,opts):
    todo=[t for t in self.todo if t[0]=="pay"]
    i=orig(self,w,p,opts)
    if todo and i is not None and (len(self.todo)<len(todo)+len([t for t in self.todo if t[0]!="pay"])):
        who=todo[0][1]
        x=w.people[who]
        r=w.perceived_need(p.name,who) or {}
        true=dict(hurt=max(min(1,(10-x.hp)/10),x.ail),fear=x.fear,hunger=x.hunger)
        log.append((opts[i]["verb"],{k:round(v,2) for k,v in true.items()},{k:round(v,2) for k,v in r.items()}))
    return i
P.Need.answer=ans
for s in (1,2,3):
    spec=dict(kind="O",world="all6s",chooser="rules",seed=s,nights=60,focal=T.FOCALS3[s%3],fenn="need")
    T.lifetime_measured(spec)
c=collections.Counter(v for v,_,_ in log)
print(len(log),c)
n=len(log)
print("true hunger>0.5 at payment:",sum(t["hunger"]>0.5 for _,t,_ in log)/n,"read hunger>0.5:",sum(r.get("hunger",0)>0.5 for _,_,r in log)/n)
print("true hurt>0.2:",sum(t["hurt"]>0.2 for _,t,_ in log)/n,"read:",sum(r.get("hurt",0)>0.2 for _,_,r in log)/n)
print("true fear>0.25:",sum(t["fear"]>0.25 for _,t,_ in log)/n)
pr=[x for x in log if x[0]=="praise"]
print("praise payments with true hunger>0.5:",sum(t["hunger"]>0.5 for _,t,_ in pr),"of",len(pr))
