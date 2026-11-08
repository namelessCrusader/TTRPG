"""Can near misses be learned? Tests on the vectors the model already makes
(plan v4, section 5). CPU only: it reads what --capture saved.

    python3 -m lab.methods.learnable report     # lab/results/methods/learnable.md
    python3 -m lab.methods.learnable selftest   # the same code on synthetic vectors

Every model is at chance when the other options only LOOK useful. That says
what the models do now, not what they could learn. These tests ask whether
the information is already in GLiNER-Decide's vectors:

PROBES (5a). A logistic regression reads one option vector and says
"acceptable or not". Three tasks: acceptable against bad, acceptable against
hand-picked near miss, unusual-but-workable against bad. Three splits:
    random    options mixed at random            (can be memorised)
    items     no item in the test fold was seen   (new things)
    needs     no need in the test fold was seen   (new jobs)
A control with shuffled labels must stay near 0.5. Reading the result: good
on "needs" = the vectors carry what makes a thing fit a job, and a head or a
fine-tune can use it; good only on "random" = memorised.

A NEW HEAD (5b). A head trained in place of GLiNER-Decide's own classifier,
on lists whose needs it never saw, then used to choose. Compared with the
model's own choice on the same lists. If it wins, the model needs only a new
head (cheap). If not, the encoder must change (a LoRA fine-tune:
lab/methods/finetune_gliner.py).

ONE HEAD OR GROUPS (5c). On the stage questions (stages.py --capture): one
head for every kind of question, against one head per kind.

BGE BASELINE. The same probes on bge-small vectors of "need: thing", which
see no list and no scene.

CONTROLS (docs/learnability-review.md, section 3.2):
    answer only    the probe on each thing's AVERAGE vector over all its
                   lists (no need in it). Close to the real probe on the new
                   needs split = it learned "generally useful things", not fit.
    memorise       each thing gets one fixed random label; the random-split
                   score of that task is how much a probe can memorise.
                   Selectivity = real score - this score.
    second source  near misses chosen a second way (the wrong options closest
                   in meaning to the need, bge-small), so a result does not
                   rest on the style of the hand-picked list.
    needs curve    the new-needs score when training on 10, 20, 30 needs: a
                   failure may mean "too few needs", not "cannot learn".
The fifth control, whether the hand-picked near misses are really wrong, needs
a person: `python3 -m lab.methods.learnable audit` writes a sample to rate.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RES = HERE.parent / "results" / "methods"


# ------------------------------------------------------------------ data
def _near_sets():
    from lab.methods.cases import NEARMISS
    return {need: {x.strip() for x in s.split(",")} for need, s in NEARMISS.items()}


def option_rows(scorer="gdecide:all", tags=(("cap_things", "cases_things.jsonl"),
                                            ("cap_creative", "cases_creative.jsonl"))):
    """One row per option: its vector, need, item, and label."""
    from lab.methods.prior import load_vectors
    near = _near_sets()
    X, need, item, lab, case, logit, pos = [], [], [], [], [], [], []
    for tag, cf in tags:
        cases = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / cf)}
        vec = load_vectors(scorer, tag)
        for cid, (V, lg) in vec.items():
            c = cases.get(cid)
            if not c:
                continue
            good = set(c["good"])
            unusual = set(c.get("unusual", []))
            for i, n in enumerate(c["names"]):
                if i in unusual:
                    l = "unusual"
                elif i in good:
                    l = "good"
                elif n in near.get(c["need"], ()):
                    l = "near"
                else:
                    l = "bad"
                X.append(V[i])
                need.append(c["need"])
                item.append(n)
                lab.append(l)
                case.append(cid)
                logit.append(float(lg[i]))
                pos.append(i)
    return {"X": np.array(X, dtype=np.float32), "need": np.array(need), "item": np.array(item),
            "label": np.array(lab), "case": np.array(case), "logit": np.array(logit), "pos": np.array(pos)}


def bge_rows(d):
    """bge-small vectors of 'need: thing' (no list, no scene), on the CPU."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    m = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").eval()
    texts = [f"{n}: {i}" for n, i in zip(d["need"], d["item"])]
    out = []
    with torch.no_grad():
        for k in range(0, len(texts), 256):
            b = tok(texts[k:k + 256], padding=True, truncation=True, return_tensors="pt")
            v = m(**b).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(v, dim=-1).numpy())
    return dict(d, X=np.concatenate(out))


# ------------------------------------------------------------------ probes
def probe(X, y, groups=None, folds=5, shuffle=False, seed=0):
    """Mean AUC over folds. groups: None = random split, else a GroupKFold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    if shuffle:
        y = rng.permutation(y)
    if len(set(y)) < 2 or min(np.bincount(y)) < folds:
        return float("nan")
    if groups is None:
        split = StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y)
    else:
        if len(set(groups)) < folds:
            return float("nan")
        split = GroupKFold(folds).split(X, y, groups)
    aucs = []
    for tr, te in split:
        if len(set(y[te])) < 2 or len(set(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function(X[te])))
    return float(np.mean(aucs)) if aucs else float("nan")


TASKS = {"acceptable vs bad": ({"good"}, {"bad"}),
         "acceptable vs near miss": ({"good", "unusual"}, {"near"}),
         "acceptable vs near miss (2nd source)": ({"good", "unusual"}, {"near2"}),
         "unusual vs bad": ({"unusual"}, {"bad"})}


def add_second_source(d, top=0.2):
    """Mark as 'near2' the wrong options that are closest in meaning to the
    need in each list (bge-small), unless already a hand-picked near miss."""
    from lab.methods.failures import similarities
    from lab.methods.cases import parse
    sim = similarities(parse()[0])
    lab = d["label"].copy()
    by = defaultdict(list)
    for k, c in enumerate(d["case"]):
        by[c].append(k)
    for c, ks in by.items():
        bad = [k for k in ks if lab[k] == "bad"]
        if not bad:
            continue
        need = d["need"][ks[0]]
        if need not in sim:
            continue
        s = sim[need]
        score = {k: max(s["verb"].get(d["item"][k], 0), s["object"].get(d["item"][k], 0)) for k in bad}
        for k in sorted(bad, key=lambda k: -score[k])[:max(1, round(top * len(bad)))]:
            lab[k] = "near2"
    return dict(d, label=lab)


def answer_only(d):
    """Each row's vector replaced by its thing's average vector over all lists."""
    acc = defaultdict(list)
    for k, it in enumerate(d["item"]):
        acc[it].append(k)
    mean = {it: d["X"][ks].mean(0) for it, ks in acc.items()}
    return dict(d, X=np.stack([mean[it] for it in d["item"]]))


def memorise_control(d, seed=0):
    """AUC on the random split when each thing has one fixed random label."""
    rng = np.random.default_rng(seed)
    items = sorted(set(d["item"]))
    lab = dict(zip(items, rng.integers(0, 2, len(items))))
    return probe(d["X"], np.array([lab[i] for i in d["item"]]))


def needs_curve(d, pos={"good", "unusual"}, neg={"near", "bad"}, sizes=(10, 20, 30), seed=0):
    """AUC on held-out needs when training on n needs."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    m = np.isin(d["label"], list(pos | neg))
    X, y, needs = d["X"][m], np.isin(d["label"][m], list(pos)).astype(int), d["need"][m]
    all_needs = sorted(set(needs))
    out = {}
    for n in sizes:
        if n >= len(all_needs):
            continue
        aucs = []
        for rep_ in range(5):
            tr_needs = set(rng.choice(all_needs, n, replace=False))
            tr = np.array([x in tr_needs for x in needs])
            if len(set(y[tr])) < 2 or len(set(y[~tr])) < 2:
                continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
            clf.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[~tr], clf.decision_function(X[~tr])))
        out[n] = float(np.mean(aucs)) if aucs else float("nan")
    return out


def audit(n=60, seed=0):
    """A sample of hand-picked near misses for a person to rate: is it really
    useless for the need, or would it work? (review section 3.2, point 3)"""
    from lab.methods.cases import NEARMISS
    rng = np.random.default_rng(seed)
    pairs = [(need, x.strip()) for need, s in NEARMISS.items() for x in s.split(",") if x.strip()]
    pick = [pairs[i] for i in rng.choice(len(pairs), min(n, len(pairs)), replace=False)]
    L = ["# Are the near misses really wrong?", "",
         "For each: would it do the job? Mark **no** (really useless), **yes** (it would work) or **maybe** "
         "(it depends on something the text does not say). A near miss that works is a wrong label, and no "
         "model can learn it from text. Built by `lab/methods/learnable.py audit`.", ""]
    L += [f"- [ ] {need}: **{item}** (yes / no / maybe)" for need, item in pick]
    (RES / "nearmiss_audit.md").write_text("\n".join(L) + "\n")
    print(RES / "nearmiss_audit.md")


def probe_table(d, name):
    L = [f"### {name}", "", "AUC (0.5 = chance, 1 = perfect). `shuffled` = the random split with the labels "
         "shuffled: it must stay near 0.5.", "",
         "| task | n | random | new items | new needs | shuffled |", "| --- | --- | --- | --- | --- | --- |"]
    for task, (pos, neg) in TASKS.items():
        m = np.isin(d["label"], list(pos | neg))
        if m.sum() < 50:
            continue
        X, y = d["X"][m], np.isin(d["label"][m], list(pos))
        cells = [probe(X, y), probe(X, y, d["item"][m]), probe(X, y, d["need"][m]), probe(X, y, shuffle=True)]
        L.append(f"| {task} | {int(m.sum())} ({int(y.sum())} +) | " + " | ".join(f"{c:.2f}" for c in cells) + " |")
    return L + [""]


# ------------------------------------------------------------------ a new head
def head_choice(d, folds=5, seed=0):
    """Train a head on lists of some needs; choose on lists of the others.
    Returns (share right with the model's own classifier, with the new head,
    with the new head plus the model's logit), on the same held-out lists."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ok = np.isin(d["label"], ["good", "unusual"]).astype(int)
    needs = d["need"]
    res = defaultdict(list)
    if len(set(needs)) < folds:
        return None
    for tr, te in GroupKFold(folds).split(d["X"], ok, needs):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
        clf.fit(d["X"][tr], ok[tr])
        s_head = clf.decision_function(d["X"][te])
        s_both = s_head + d["logit"][te]
        by = defaultdict(list)
        for j, k in enumerate(te):
            by[d["case"][k]].append((j, k))
        for cid, idx in by.items():
            js = [j for j, _ in idx]
            ks = [k for _, k in idx]
            if not ok[ks].any():
                continue
            res["model"].append(ok[ks][int(np.argmax(d["logit"][ks]))])
            res["head"].append(ok[ks][int(np.argmax(s_head[js]))])
            res["head + model"].append(ok[ks][int(np.argmax(s_both[js]))])
    return {k: (float(np.mean(v)), len(v)) for k, v in res.items()}


# ------------------------------------------------------------------ a list-wise head (to-do A1)
def _segments(case):
    """Row indices grouped by list, and each row's list number."""
    ids, seg = np.unique(case, return_inverse=True)
    return len(ids), seg


def _list_logsoftmax(s, seg, n):
    """log softmax of the scores s within each list (seg = list number)."""
    import torch
    mx = torch.full((n,), -1e30).scatter_reduce(0, seg, s, "amax", include_self=True)
    z = s - mx[seg].detach()
    tot = torch.zeros(n).index_add(0, seg, z.exp())
    return z - tot.log()[seg]


def _fit_list_head(F, ok, seg, n, hidden, epochs=300, lr=3e-3, wd=1e-2, seed=0):
    """Train on whole lists: loss = -log(total probability on the acceptable
    options of each list). Full batch, Adam."""
    import torch
    torch.manual_seed(seed)
    F = torch.tensor(F)
    seg_t = torch.tensor(seg)
    okm = torch.tensor(ok, dtype=torch.bool)
    has = torch.zeros(n, dtype=torch.bool).index_fill(0, seg_t[okm], True)
    if hidden:
        net = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(F.shape[1], hidden), torch.nn.GELU(),
                                  torch.nn.Linear(hidden, 1))
    else:
        net = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(F.shape[1], 1))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        net.train()
        lp = _list_logsoftmax(net(F).squeeze(-1), seg_t, n)
        acc = torch.zeros(n).index_add(0, seg_t, torch.where(okm, lp.exp(), torch.zeros_like(lp)))
        loss = -(acc[has].clamp_min(1e-9).log()).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    return net


def list_features(d, answer_only_vec=False):
    """Per option: its vector (or, for the control, the thing's average vector
    over all lists), its difference from its list's mean, and GLiNER's logit."""
    X = d["X"]
    if answer_only_vec:
        X = answer_only(d)["X"]
    n, seg = _segments(d["case"])
    mean = np.zeros((n, X.shape[1]), dtype=np.float32)
    np.add.at(mean, seg, X)
    mean /= np.bincount(seg, minlength=n)[:, None]
    return np.hstack([X, X - mean[seg], d["logit"][:, None]]).astype(np.float32)


def listwise(d, folds=5, seed=0):
    """Top pick on lists of needs never seen in training: GLiNER's own choice,
    the per-option head, and heads trained on whole lists. Lists that hold a
    near miss are also reported apart (right, and how often the pick is the
    near miss). Control: the same list head on each thing's average vector
    (no need in it); if it does as well, the head learned 'useful things',
    not fit."""
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    torch.set_num_threads(4)
    ok = np.isin(d["label"], ["good", "unusual"])
    near = d["label"] == "near"
    feats = {"full": list_features(d), "answer only": list_features(d, answer_only_vec=True)}
    arms = {"model": None, "per-option head": None, "list head, linear": ("full", 0),
            "list head, 1 hidden layer": ("full", 128), "list head, no GLiNER score": ("nologit", 128),
            "control: answer only": ("answer only", 128)}
    res = defaultdict(list)
    res_near = defaultdict(list)
    res_nearpick = defaultdict(list)
    for tr, te in GroupKFold(folds).split(d["X"], ok, d["need"]):
        n_tr, seg_tr = _segments(d["case"][tr])
        n_te, seg_te = _segments(d["case"][te])
        scores = {"model": d["logit"][te]}
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
        clf.fit(d["X"][tr], ok[tr].astype(int))
        scores["per-option head"] = clf.decision_function(d["X"][te])
        for arm, spec in arms.items():
            if spec is None:
                continue
            key, hidden = spec
            F = feats["full"] if key in ("full", "nologit") else feats[key]
            if key != "full":                    # the controls see no GLiNER score
                F = F[:, :-1]
            mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
            net = _fit_list_head((F[tr] - mu) / sd, ok[tr], seg_tr, n_tr, hidden, seed=seed)
            with torch.no_grad():
                scores[arm] = net(torch.tensor((F[te] - mu) / sd)).squeeze(-1).numpy()
        for k in range(n_te):
            rows = np.where(seg_te == k)[0]
            idx = te[rows]
            if not ok[idx].any():
                continue
            has_near = near[idx].any()
            for arm, s in scores.items():
                j = idx[int(np.argmax(s[rows]))]
                res[arm].append(ok[j])
                if has_near:
                    res_near[arm].append(ok[j])
                    res_nearpick[arm].append(near[j])
    return {arm: (float(np.mean(res[arm])), len(res[arm]),
                  float(np.mean(res_near[arm])) if res_near[arm] else float("nan"),
                  float(np.mean(res_nearpick[arm])) if res_nearpick[arm] else float("nan"), len(res_near[arm]))
            for arm in res}


# ------------------------------------------------------------------ one head or groups
def stage_rows(scorer="gdecide:all"):
    """The stage questions' label vectors with right/wrong, from stages.py
    --capture and its records."""
    root = RES / "stages"
    X, task, right, case, need, logit = [], [], [], [], [], []
    case_files = {"capability": "cases_things.jsonl", "people": "cases_people.jsonl"}
    for recf in sorted(root.glob(f"*__{scorer.replace(':', '__')}.jsonl")):
        exp = recf.stem.split("__")[0]
        if exp not in case_files:
            continue
        needs = {json.loads(l)["id"]: json.loads(l)["need"] for l in open(HERE / case_files[exp])}
        recs = {json.loads(l)["id"]: json.loads(l) for l in open(recf)}
        capd = recf.with_suffix(".cap")
        for part in sorted(capd.glob("part*.npz")) if capd.exists() else []:
            z = np.load(part)
            off = 0
            for cid, q, labs, n in zip(z["ids"], z["question"], z["labels"], z["n"]):
                V = z["opts"][off:off + n].astype(np.float32)
                lg = z["logits"][off:off + n]
                off += n
                r = recs.get(str(cid))
                if not r or "stage" not in r or str(q).startswith("FINAL"):
                    continue
                labels = str(labs).split("␟")
                st = r["stage"]
                if exp == "capability" and q.startswith("What does Kael need"):
                    good = set(st.get("right", []))
                    t = "capability"
                elif exp == "people" and q.startswith("What does this situation call for"):
                    good = set(st.get("right_calls", []))
                    t = "calls for"
                else:
                    continue
                for v, lab, g in zip(V, labels, lg):
                    X.append(v)
                    task.append(t)
                    right.append(lab in good)
                    case.append(f"{t}|{cid}")
                    need.append(needs[str(cid)])
                    logit.append(float(g))
    return {"X": np.array(X, dtype=np.float32), "task": np.array(task), "right": np.array(right),
            "case": np.array(case), "need": np.array(need), "logit": np.array(logit)}


def group_heads(d, folds=5):
    """Per task, on needs never seen in training: AUC of one head trained on
    every task, and of one head per task; and the top pick of each question
    (the head's against GLiNER's own). Split by NEED: each need is asked many
    times, so a split by case lets a head memorise the need's answer."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out = defaultdict(lambda: defaultdict(list))
    y = d["right"].astype(int)
    groups = d.get("need", d["case"])
    for tr, te in GroupKFold(folds).split(d["X"], y, groups):
        shared = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
        shared.fit(d["X"][tr], y[tr])
        for t in set(d["task"]):
            trt = tr[d["task"][tr] == t]
            tet = te[d["task"][te] == t]
            if len(set(y[tet])) < 2 or len(set(y[trt])) < 2:
                continue
            own = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
            own.fit(d["X"][trt], y[trt])
            s_own = own.decision_function(d["X"][tet])
            out[t]["one head for all"].append(roc_auc_score(y[tet], shared.decision_function(d["X"][tet])))
            out[t]["its own head"].append(roc_auc_score(y[tet], s_own))
            if "logit" in d:
                by = defaultdict(list)
                for j, k in enumerate(tet):
                    by[d["case"][k]].append((j, k))
                for idx in by.values():
                    js, ks = [j for j, _ in idx], [k for _, k in idx]
                    out[t]["top pick, head"].append(y[ks][int(np.argmax(s_own[js]))])
                    out[t]["top pick, GLiNER"].append(y[ks][int(np.argmax(d["logit"][ks]))])
    return {t: {k: float(np.mean(v)) for k, v in r.items()} for t, r in out.items()}


# ------------------------------------------------------------------ report
def report():
    L = ["# Can near misses be learned?", "",
         "Tests on the vectors GLiNER-Decide already makes (`--capture`). Built by `lab/methods/learnable.py`.", ""]
    d = option_rows()
    if not len(d["X"]):
        L += ["No captured vectors yet: run the capture pass in `lab/methods/v4_queue.sh` first."]
    else:
        d = add_second_source(d)
        L += ["## Probes", ""] + probe_table(d, "GLiNER-Decide option vectors (in their list and scene)")
        L += probe_table(answer_only(d), "Control: answer only (each thing's average vector, no need)")
        L += probe_table(bge_rows(d), "bge-small, 'need: thing' alone (no list, no scene)")
        mem = memorise_control(d)
        curve = needs_curve(d)
        L += [f"Control, memorise: each thing given one fixed random label gets AUC {mem:.2f} on the random "
              "split. Selectivity = the real random-split AUC minus this.", "",
              "New-needs AUC (acceptable against near miss or bad) when training on n needs: "
              + ", ".join(f"{n} needs {v:.2f}" for n, v in curve.items()) + ".", ""]
        h = head_choice(d)
        if h:
            L += ["## A new head, on needs it never saw", "",
                  "| chooser | share right | lists |", "| --- | --- | --- |"]
            L += [f"| {k} | {v[0]:.0%} | {v[1]} |" for k, v in h.items()] + [""]
        lw = listwise(d)
        L += ["## A head that sees the whole list (to-do A1)", "",
              "Trained on whole lists (softmax over the list), on needs it never saw (5 folds by need). "
              "`near lists` = the lists that hold a hand-picked near miss; `picks the near miss` = how often the "
              "pick is that near miss. The control sees each thing's average vector (no need in it) and no "
              "GLiNER score: if it does as well, the head learned 'useful things', not fit.", "",
              "| chooser | share right | lists | near lists: right | picks the near miss | near lists |",
              "| --- | --- | --- | --- | --- | --- |"]
        L += [f"| {k} | {v[0]:.0%} | {v[1]} | {v[2]:.0%} | {v[3]:.0%} | {v[4]} |" for k, v in lw.items()] + [""]
    s = stage_rows()
    if len(s["X"]):
        g = group_heads(s)
        L += ["## One head, or one per kind of question", "",
              "On needs never seen in training (5 folds by need). AUC of the right labels against the wrong ones; "
              "`top pick` = the share of questions whose top label is right.", "",
              "| question | AUC, one head for all | AUC, its own head | top pick, its own head | top pick, GLiNER |",
              "| --- | --- | --- | --- | --- |"]
        f = lambda r, k, pct=False: (f"{r[k]:.0%}" if pct else f"{r[k]:.2f}") if k in r else "-"
        L += [f"| {t} | {f(r, 'one head for all')} | {f(r, 'its own head')} | {f(r, 'top pick, head', 1)} | "
              f"{f(r, 'top pick, GLiNER', 1)} |" for t, r in g.items()] + [""]
    (RES / "learnable.md").write_text("\n".join(L) + "\n")
    print(RES / "learnable.md")


def selftest(seed=0):
    """Synthetic vectors: a 'fits the job' direction that depends on the
    need, plus an item identity part. The probe must find the signal on new
    needs, and the shuffled control must stay near 0.5."""
    rng = np.random.default_rng(seed)
    dim, needs, items = 64, [f"n{i}" for i in range(20)], [f"i{j}" for j in range(80)]
    fit_dir = rng.normal(size=dim)
    item_vec = {i: rng.normal(size=dim) for i in items}
    X, need, item, lab, case, logit = [], [], [], [], [], []
    for k in range(400):
        n = needs[k % 20]
        its = rng.choice(items, 16, replace=False)
        good = int(rng.integers(16))
        for j, it in enumerate(its):
            v = item_vec[it] + rng.normal(scale=0.5, size=dim) + (1.5 * fit_dir if j == good else 0)
            X.append(v)
            need.append(n)
            item.append(it)
            lab.append("good" if j == good else "bad")
            case.append(str(k))
            logit.append(float(rng.normal()))          # a model that has not learned it
    d = {"X": np.array(X), "need": np.array(need), "item": np.array(item), "label": np.array(lab),
         "case": np.array(case), "logit": np.array(logit)}
    y = d["label"] == "good"
    a_need, a_shuf = probe(d["X"], y, d["need"]), probe(d["X"], y, shuffle=True)
    print(f"  probe on new needs {a_need:.2f}, shuffled {a_shuf:.2f}")
    assert a_need > 0.8 and abs(a_shuf - 0.5) < 0.1
    h = head_choice(d)
    print("  choice:", {k: f"{v[0]:.0%}" for k, v in h.items()})
    assert h["head"][0] > h["model"][0] + 0.3
    lw = listwise(d)
    print("  list heads:", {k: f"{v[0]:.0%}" for k, v in lw.items()})
    assert lw["list head, linear"][0] > lw["model"][0] + 0.3
    assert lw["control: answer only"][0] < lw["list head, linear"][0] - 0.2
    ao = probe(answer_only(d)["X"], y, d["need"])
    print(f"  answer only on new needs {ao:.2f} (the planted signal is not in the thing alone)")
    assert ao < a_need - 0.2
    print(f"  memorise control {memorise_control(d):.2f}; needs curve {needs_curve(d, {'good'}, {'bad'})}")
    s = {"X": d["X"], "task": np.where(np.arange(len(y)) % 2, "a", "b"), "right": y, "case": d["case"]}
    print("  groups:", group_heads(s))
    print("selftest passed")


if __name__ == "__main__":
    {"report": report, "selftest": selftest, "audit": audit}[sys.argv[1]]()
