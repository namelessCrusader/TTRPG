"""Every model behind one call: decide(state, questions) -> answers.

The questions are Laya's typed form, because it is the smallest thing that
says what we need and three of the models already speak it:

    choice  pick one key of `criteria` (a dict of key -> what it means)
    score   pick an index into `criteria` (a list, lowest first)
    noul    yes or no, with a probability

and every answer comes back as {"answer": ..., "p": ..., "p_true": ...} so the
scorer never has to know whose it was. A model that cannot give a probability
leaves it out; it is not invented for it.

Nothing here imports the simulation. The cards are the only thing that crosses.
"""
from __future__ import annotations

import math


def state_text(state: dict) -> str:
    """Key: value lines, keys in a fixed order — the same bytes every run."""
    return "\n".join(f"{k}: {state[k]}" for k in sorted(state))


def _labels(q):
    c = q["criteria"]
    return list(c) if isinstance(c, dict) else list(range(len(c)))


def _describe(q, key):
    c = q["criteria"]
    return f"{key.replace('_', ' ')}: {c[key]}" if isinstance(c, dict) else c[key]


def _vram():
    try:
        import torch
        return torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
    except ImportError:
        return 0.0


class Model:
    def load(self): ...

    def decide(self, state: str, questions: dict) -> dict: ...

    def vram_mb(self) -> float:
        return _vram()


class Majority(Model):
    """Ignores the state. The line every other model has to clear."""
    def decide(self, state, questions):
        out = {}
        for name, q in questions.items():
            if q["type"] == "noul":
                out[name] = {"answer": True, "p_true": 0.5}
            else:
                out[name] = {"answer": _labels(q)[0] if q["type"] == "choice" else 0}
        return out


class Laya(Model):
    """convaiinnovations/laya — ModernBERT-large + option-marker head, Apache-2.0.
    Takes the typed questions natively, all of them in one pass."""
    repo = "convaiinnovations/laya"

    def load(self):
        import laya
        self.m = laya.load(self.repo)

    def decide(self, state, questions):
        res = self.m.predict({"text": state}, questions)
        # what it was actually shown: the English checkpoint reads 512 tokens,
        # and past that the input is cut without a word
        self.tokens = (res.get("usage") or {}).get("input_tokens")
        return {n: _typed(questions[n], a) for n, a in res["answers"].items()}


class Von(Model):
    """wfzyx/von — ModernBERT-large, Apache-2.0, the other open Jev."""
    def load(self):
        import von
        self.von = von

    def decide(self, state, questions):
        v = self.von
        qs = {}
        for n, q in questions.items():
            if q["type"] == "choice":
                qs[n] = v.choice(instructions=q["instructions"], criteria=q["criteria"])
            elif q["type"] == "score":
                qs[n] = v.score(instructions=q["instructions"], criteria=q["criteria"])
            else:
                qs[n] = v.noul(instructions=q["instructions"])
        res = v.system_one(state={"text": state}, questions=qs)
        return {n: _typed(questions[n], a) for n, a in _answers(res).items()}


def _answers(res):
    if isinstance(res, dict):
        return res.get("answers", res)
    return getattr(res, "answers", res)


def _get(a, *keys):
    for k in keys:
        if isinstance(a, dict) and k in a:
            return a[k]
        if hasattr(a, k):
            return getattr(a, k)
    return None


def _typed(q, a):
    """Laya's and Von's answer objects, into ours.

    Laya's `noul` IS the probability of yes (its `confidence` is something
    else, |2p-1|-shaped — read as p it flips every answer), and its `score` is
    an expectation over the rubric; the answer is the rubric's likeliest step."""
    probs = _get(a, "probabilities")
    if q["type"] == "choice":
        ch = _get(a, "choice", "answer")
        return {"answer": ch, "p": (probs or {}).get(ch) if isinstance(probs, dict)
                else _get(a, "confidence")}
    if q["type"] == "score":
        if isinstance(probs, dict) and probs:
            best = max(probs, key=probs.get)
            return {"answer": int(best), "p": probs[best],
                    "expected": _get(a, "score")}
        s = _get(a, "score", "answer", "choice")
        if isinstance(s, str) and s in q["criteria"]:
            s = q["criteria"].index(s)
        return {"answer": None if s is None else int(round(float(s)))}
    p = _get(a, "noul", "p_true", "probability")
    if isinstance(p, bool):
        return {"answer": p, "p_true": None}
    if isinstance(p, str):
        p = {"yes": 1.0, "true": 1.0, "no": 0.0, "false": 0.0}.get(p.lower())
    return {"answer": None if p is None else p >= 0.5, "p_true": p}


class GLiClass(Model):
    """knowledgator GLiClass — labels ride in the input, one pass, Apache-2.0.
    It has no question slot, so the question is written above the state, and
    yes/no is two labels."""
    repo = "knowledgator/gliclass-modern-base-v3.0"

    def load(self):
        import torch
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline
        from transformers import AutoTokenizer
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        m = GLiClassModel.from_pretrained(self.repo)
        tok = AutoTokenizer.from_pretrained(self.repo, add_prefix_space=True)
        self.pipe = ZeroShotClassificationPipeline(
            m, tok, classification_type="single-label", device=dev)

    def decide(self, state, questions):
        out = {}
        for n, q in questions.items():
            text = f"{q['instructions']}\n{state}"
            if q["type"] == "noul":
                labels = {"yes": True, "no": False}
            else:
                labels = {_describe(q, k): k for k in _labels(q)}
            got = self.pipe(text, list(labels), threshold=0.0)[0]
            s = {labels[g["label"]]: g["score"] for g in got}
            best = max(s, key=s.get)
            if q["type"] == "noul":
                pt = s.get(True, 0.0) / max(1e-9, s.get(True, 0.0) + s.get(False, 0.0))
                out[n] = {"answer": pt >= 0.5, "p_true": pt}
            else:
                out[n] = {"answer": best, "p": s[best]}
        return out


class QwenPick(Model):
    """A small decoder used as a picker: the mean log-probability of each
    option's words after the question. What src/core/llm.py's ChoiceLM does,
    rewritten here so the lab does not import the engine. One forward pass per
    option — the cost the encoders exist to avoid, and the thing to beat."""
    repo = "Qwen/Qwen3-0.6B"

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(self.repo)
        self.lm = AutoModelForCausalLM.from_pretrained(
            self.repo, torch_dtype=torch.float16 if self.dev == "cuda" else torch.float32
        ).to(self.dev).eval()

    def _logp(self, prompt, conts):
        torch = self.torch
        p_ids = self.tok(prompt, return_tensors="pt").input_ids[0]
        scores = []
        for c in conts:
            c_ids = self.tok(" " + c, return_tensors="pt").input_ids[0]
            ids = torch.cat([p_ids, c_ids]).unsqueeze(0).to(self.dev)
            with torch.no_grad():
                logits = self.lm(ids).logits[0, len(p_ids) - 1:-1].float()
            lp = torch.log_softmax(logits, -1).gather(1, c_ids.to(self.dev).unsqueeze(1))
            scores.append(lp.mean().item())
        return scores

    def decide(self, state, questions):
        out = {}
        for n, q in questions.items():
            if q["type"] == "noul":
                keys, texts = [True, False], ["yes", "no"]
            else:
                keys = _labels(q)
                texts = [_describe(q, k) for k in keys]
            menu = "\n".join(f"- {t}" for t in texts)
            prompt = (f"{state}\n\nQuestion: {q['instructions']}\n"
                      f"Options:\n{menu}\nAnswer:")
            lp = self._logp(prompt, texts)
            z = max(lp)
            w = [math.exp(x - z) for x in lp]
            p = [x / sum(w) for x in w]
            i = max(range(len(p)), key=p.__getitem__)
            if q["type"] == "noul":
                out[n] = {"answer": keys[i], "p_true": p[0]}
            else:
                out[n] = {"answer": keys[i], "p": p[i]}
        return out


MODELS = {"majority": Majority, "laya": Laya, "von": Von,
          "gliclass": GLiClass, "qwen": QwenPick}
