"""Every model under every method, behind one call.

    scorer = make("laya_en:one")
    scorer.load()
    out = scorer.rank(situation, question, options)
    # out = {"scores": [...] or None, "pick": index, "order": [indices, best first],
    #        "prob": True if the scores are probabilities over the options}

METHODS (the same names as the plan, Workstream A):

    all    all options in one input, scored together
    one    one input for each option, scored alone
    two    two options at a time; the winner goes on (a knockout)
    apart  the situation and each option are read separately and compared
    short  "apart" keeps the best 10, then "one" chooses among them

Models are loaded one at a time by the runner, which checks GPU memory first.
"""
from __future__ import annotations

import copy
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch  # noqa: E402

torch.set_num_threads(4)


def _softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [x / s for x in e]


def _order(scores):
    return sorted(range(len(scores)), key=lambda i: -scores[i])


def knockout(n, compare):
    """Two at a time. `compare(pairs) -> [p(first wins)]` for a batch of pairs.
    Returns the winner and an order: later eliminations rank higher."""
    alive, out_round = list(range(n)), {}
    rnd = 0
    while len(alive) > 1:
        pairs = [(alive[i], alive[i + 1]) for i in range(0, len(alive) - 1, 2)]
        p = compare(pairs)
        nxt = [a if pa >= 0.5 else b for (a, b), pa in zip(pairs, p)]
        for (a, b), w in zip(pairs, nxt):
            out_round[b if w == a else a] = rnd
        if len(alive) % 2:
            nxt.append(alive[-1])
        alive, rnd = nxt, rnd + 1
    out_round[alive[0]] = rnd
    order = sorted(range(n), key=lambda i: -out_round[i])
    return {"scores": None, "pick": alive[0], "order": order, "prob": False}


# ---------------------------------------------------------------- Laya family
class Laya:
    """convaiinnovations/laya. English (421M, 512 word pieces) or the
    multilingual checkpoint (322M, 1,024)."""
    def __init__(self, method, sub=None):
        self.method, self.sub = method, sub

    params = 0.45e9

    def load(self):
        import laya
        self.a = laya.load("convaiinnovations/laya", subfolder=self.sub)

    def _predict(self, state, qs, chunk=16):
        out, keys = {}, list(qs)
        for i in range(0, len(keys), chunk):
            part = {k: qs[k] for k in keys[i:i + chunk]}
            out.update(self.a.predict(state, part)["answers"])
        return out

    def rank(self, sit, question, options):
        m = self.method
        if m == "all":
            q = {"q": {"type": "choice", "instructions": question,
                       "criteria": {o: None for o in options}}}
            pr = self.a.predict(sit, q)["answers"]["q"]["probabilities"]
            sc = [pr[o] for o in options]
            return {"scores": sc, "pick": max(range(len(sc)), key=sc.__getitem__),
                    "order": _order(sc), "prob": True}
        if m == "one":
            qs = {f"o{i}": {"type": "noul",
                            "instructions": f"{question} Proposed answer: {o}. Is this a good answer?"}
                  for i, o in enumerate(options)}
            ans = self._predict(sit, qs)
            sc = [ans[f"o{i}"]["noul"] for i in range(len(options))]
            return {"scores": sc, "pick": max(range(len(sc)), key=sc.__getitem__),
                    "order": _order(sc), "prob": False}
        if m == "two":
            def compare(pairs):
                qs = {}
                for j, (a, b) in enumerate(pairs):     # both orders: position must not decide
                    qs[f"f{j}"] = {"type": "choice", "instructions": question,
                                   "criteria": {options[a]: None, options[b]: None}}
                    qs[f"r{j}"] = {"type": "choice", "instructions": question,
                                   "criteria": {options[b]: None, options[a]: None}}
                ans = self._predict(sit, qs)
                return [0.5 * (ans[f"f{j}"]["probabilities"][options[a]]
                               + ans[f"r{j}"]["probabilities"][options[a]])
                        for j, (a, b) in enumerate(pairs)]
            return knockout(len(options), compare)
        if m == "solo":
            # EACH OPTION READ ALONE, NOT AS YES/NO: one option per reading,
            # and Laya's raw score for it (before the comparison between
            # options). All readings go through the GPU as one batch.
            sc = self._raw_scores(sit, question, options)
            return {"scores": sc, "pick": max(range(len(sc)), key=sc.__getitem__),
                    "order": _order(sc), "prob": False}
        if m == "anchor":
            # each option against ONE fixed alternative
            none = "none of these things"
            qs = {f"o{i}": {"type": "choice", "instructions": question,
                            "criteria": {o: None, none: None}} for i, o in enumerate(options)}
            ans = self._predict(sit, qs)
            sc = [ans[f"o{i}"]["probabilities"][o] for i, o in enumerate(options)]
            return {"scores": sc, "pick": max(range(len(sc)), key=sc.__getitem__),
                    "order": _order(sc), "prob": False}
        if m == "apart":
            q = self._embed([f"{sit} {question}"])[0]
            e = self._embed(options)
            sc = [float(q @ v) for v in e]
            return {"scores": sc, "pick": max(range(len(sc)), key=sc.__getitem__),
                    "order": _order(sc), "prob": False}
        raise ValueError(m)

    @torch.no_grad()
    def _raw_scores(self, sit, question, options, bs=32):
        from laya.common import QTYPES, build_sequence, collate_items
        cfg, out = self.a.cfg, []
        items = []
        for o in options:
            q = {"t": "choice", "ins": question, "crit": {o: None}}
            ids, marks = build_sequence(self.a.tok, sit, q, cfg["max_len"], cfg["head_max_len"])
            items.append({"ids": ids, "markers": marks, "qtype": QTYPES["choice"]})
        for i in range(0, len(items), bs):
            b = collate_items([items[i:i + bs]], self.a.tok.pad_token_id)
            dev = self.a.device
            with torch.autocast(device_type=dev.type, dtype=self.a.dtype, enabled=dev.type == "cuda"):
                logits, _ = self.a.model(b["input_ids"].to(dev), b["attention_mask"].to(dev),
                                         b["marker_pos"].to(dev), b["marker_mask"].to(dev),
                                         b["qtype"].to(dev))
            out += [float(x) for x in logits[:, 0].float().cpu()]
        return out

    @torch.no_grad()
    def _embed(self, texts):
        tok, enc = self.a.tok, self.a.model.encoder
        b = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        b = {k: v.to(self.a.device) for k, v in b.items()}
        h = enc(**b).last_hidden_state.float()
        m = b["attention_mask"].unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1)
        return torch.nn.functional.normalize(v, dim=-1).cpu()


class Von:
    """wfzyx/von (ModernBERT-large, 395M)."""
    params = 0.45e9

    def __init__(self, method):
        self.method = method

    def load(self):
        import von
        self.v = von
        self.v.system_one(state="warm up", questions={"q": von.noul(instructions="Is this a test?")})

    def rank(self, sit, question, options):
        v = self.v
        if self.method == "all":
            r = v.system_one(state=sit, questions={
                "q": v.choice(instructions=question, criteria={o: None for o in options})})
            pr = r.answers["q"].probabilities
            sc = [pr[o] for o in options]
            return {"scores": sc, "pick": max(range(len(sc)), key=sc.__getitem__),
                    "order": _order(sc), "prob": True}
        if self.method == "one":
            sc = []
            for i in range(0, len(options), 16):
                qs = {f"o{j}": v.noul(instructions=f"{question} Proposed answer: {o}. Is this a good answer?")
                      for j, o in enumerate(options[i:i + 16])}
                r = v.system_one(state=sit, questions=qs)
                sc += [r.answers[f"o{j}"].noul for j in range(len(qs))]
            return {"scores": sc, "pick": max(range(len(sc)), key=sc.__getitem__),
                    "order": _order(sc), "prob": False}
        raise ValueError(self.method)


# ---------------------------------------------------------------- decoder LMs
class Qwen:
    """A causal language model. The situation is read ONE time and the reading
    is kept (the warm start); each option then costs only its own tokens."""
    def __init__(self, repo, method):
        self.repo, self.method = repo, method
        self.params = 1.7e9 if "1.7B" in repo else 0.6e9

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(self.repo)
        self.m = AutoModelForCausalLM.from_pretrained(self.repo, dtype=torch.bfloat16).cuda().eval()

    @torch.no_grad()
    def cont_logprobs(self, prompt, conts, bs=24):
        """Sum of log-probabilities of each continuation after `prompt`."""
        tok = self.tok
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        out = self.m(input_ids=torch.tensor([p_ids]).cuda(), use_cache=True)
        cache, first = out.past_key_values, torch.log_softmax(out.logits[0, -1].float(), -1)
        c_ids = [tok(c, add_special_tokens=False)["input_ids"] for c in conts]
        # every row of a batch carries its own copy of the kept reading, so a
        # long prompt means a small batch: 128 numbered options at 24 a batch
        # took the 0.6B model to 5.3 of 5.6 GB
        bs = max(1, min(bs, int(bs * 250 / max(len(p_ids), 1))))
        res = []
        for i in range(0, len(c_ids), bs):
            chunk = c_ids[i:i + bs]
            L = max(len(c) for c in chunk)
            ids = torch.full((len(chunk), L), tok.pad_token_id or 0, dtype=torch.long)
            att = torch.zeros((len(chunk), len(p_ids) + L), dtype=torch.long)
            att[:, :len(p_ids)] = 1
            for j, c in enumerate(chunk):
                ids[j, :len(c)] = torch.tensor(c)
                att[j, len(p_ids):len(p_ids) + len(c)] = 1
            cb = copy.deepcopy(cache)
            cb.batch_repeat_interleave(len(chunk))
            o = self.m(input_ids=ids.cuda(), attention_mask=att.cuda(), past_key_values=cb, use_cache=True)
            lp = torch.log_softmax(o.logits.float(), -1)
            for j, c in enumerate(chunk):
                s = float(first[c[0]])
                for t in range(1, len(c)):
                    s += float(lp[j, t - 1, c[t]])
                res.append(s)
        return res

    @torch.no_grad()
    def next_logprobs(self, prompts, words, bs=16):
        """For each prompt, the log-probability of each one-token word next."""
        tok = self.tok
        ids = [tok(w, add_special_tokens=False)["input_ids"] for w in words]
        assert all(len(i) == 1 for i in ids), words
        ids = [i[0] for i in ids]
        enc = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
        out = []
        for i in range(0, len(enc), bs):
            chunk = enc[i:i + bs]
            L = max(len(e) for e in chunk)
            x = torch.full((len(chunk), L), tok.pad_token_id or 0, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for j, e in enumerate(chunk):            # left padding: the last position is real
                x[j, L - len(e):] = torch.tensor(e)
                att[j, L - len(e):] = 1
            h = self.m.model(input_ids=x.cuda(), attention_mask=att.cuda()).last_hidden_state[:, -1]
            lp = torch.log_softmax(self.m.lm_head(h).float(), -1)
            out += [[float(lp[j, t]) for t in ids] for j in range(len(chunk))]
        return out

    @torch.no_grad()
    def plain_logprobs(self, prompt, conts):
        """The same without the kept reading: the check that the cache is right."""
        tok = self.tok
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        res = []
        for c in conts:
            ids = p_ids + tok(c, add_special_tokens=False)["input_ids"]
            lp = torch.log_softmax(self.m(input_ids=torch.tensor([ids]).cuda()).logits[0].float(), -1)
            res.append(sum(float(lp[t - 1, ids[t]]) for t in range(len(p_ids), len(ids))))
        return res

    def rank(self, sit, question, options):
        m = self.method
        if m in ("cloze", "cloze_cal"):
            # A SENTENCE TO COMPLETE instead of question and answer: "... He
            # takes the kitchen knife." cloze_cal divides out how likely the
            # words are after "He takes" with no situation.
            conts = [f" {o}." for o in options]
            sc = self.cont_logprobs(f"{sit} He takes", conts)
            if m == "cloze_cal":
                base = self.cont_logprobs("He takes", conts)
                sc = [x - y for x, y in zip(sc, base)]
            return {"scores": sc, "pick": max(range(len(sc)), key=sc.__getitem__),
                    "order": _order(sc), "prob": m == "cloze"}
        if m in ("one", "one_mean", "one_cal"):
            # THREE SCORE RULES, because the plain total favors short names:
            # "the wooden spoon." outscored "the kitchen knife." for cutting a
            # rope, by length alone.
            #   one       total log-probability of the option (with its end mark)
            #   one_mean  the same, per token
            #   one_cal   the total minus the total with no situation at all
            #             (how likely the words are anyway, divided out)
            conts = [f" {o}." for o in options]
            sc = self.cont_logprobs(f"{sit}\nQuestion: {question}\nAnswer:", conts)
            if m == "one_mean":
                sc = [x / len(self.tok(c, add_special_tokens=False)["input_ids"])
                      for x, c in zip(sc, conts)]
            if m == "one_cal":
                base = self.cont_logprobs(f"Question: {question}\nAnswer:", conts)
                sc = [x - y for x, y in zip(sc, base)]
            return {"scores": sc, "pick": max(range(len(sc)), key=sc.__getitem__),
                    "order": _order(sc), "prob": m == "one"}
        if m == "all":
            listing = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(options))
            sc = self.cont_logprobs(f"{sit}\nQuestion: {question}\nOptions:\n{listing}\n"
                                    f"Answer with the number of one good option.\nAnswer:",
                                    [f" {i + 1}." for i in range(len(options))])
            return {"scores": sc, "pick": max(range(len(sc)), key=sc.__getitem__),
                    "order": _order(sc), "prob": True}
        if m == "two":
            def compare(pairs):
                # both orders for each pair, all prompts of a round in batches
                prompts = []
                for a, b in pairs:
                    for x, y in ((a, b), (b, a)):
                        prompts.append(f"{sit}\nQuestion: {question}\nA. {options[x]}\nB. {options[y]}\n"
                                       f"Which is the better answer, A or B?\nAnswer:")
                lp = self.next_logprobs(prompts, [" A", " B"])
                out = []
                for j in range(len(pairs)):
                    f, r = _softmax(lp[2 * j]), _softmax(lp[2 * j + 1])
                    out.append(0.5 * (f[0] + r[1]))   # a is A in the first, B in the second
                return out
            return knockout(len(options), compare)
        raise ValueError(m)


# ---------------------------------------------------------------- read apart
class Embed:
    """A small sentence-matching model: situation and option read apart,
    compared by cosine. Option readings are kept between calls."""
    params = 0.04e9
    repo = "BAAI/bge-small-en-v1.5"

    def __init__(self):
        self.cache = {}

    def load(self):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(self.repo)
        self.m = AutoModel.from_pretrained(self.repo).cuda().eval()

    @torch.no_grad()
    def embed(self, texts):
        new = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for i in range(0, len(new), 64):
            b = self.tok(new[i:i + 64], return_tensors="pt", padding=True, truncation=True,
                         max_length=256).to("cuda")
            v = torch.nn.functional.normalize(self.m(**b).last_hidden_state[:, 0].float(), dim=-1).cpu()
            for t, e in zip(new[i:i + 64], v):
                self.cache[t] = e
        return [self.cache[t] for t in texts]

    def rank(self, sit, question, options):
        q = self.embed([f"Represent this sentence for searching relevant passages: {sit} {question}"])[0]
        sc = [float(q @ v) for v in self.embed(options)]
        return {"scores": sc, "pick": max(range(len(sc)), key=sc.__getitem__),
                "order": _order(sc), "prob": False}


class Shortlist:
    """`apart` keeps the best `k`; `one` chooses among them."""
    def __init__(self, first, second, k=10):
        self.first, self.second, self.k = first, second, k
        self.params = first.params + second.params

    def load(self):
        self.first.load()
        self.second.load()

    def rank(self, sit, question, options):
        keep = self.first.rank(sit, question, options)["order"][:self.k]
        r = self.second.rank(sit, question, [options[i] for i in keep])
        order = [keep[i] for i in r["order"]] + [i for i in range(len(options)) if i not in keep]
        return {"scores": None, "pick": order[0], "order": order, "prob": False}


class GlinerDecide:
    """fastino/GLiNER2.5-Decide (DeBERTa-v3-large, 340M, Apache-2.0), or the
    1B version. All options in one reading, like Laya "all"; the question goes
    in as the task's instruction. Scores are the model's own per-label
    probabilities (softmax over an exclusive, pick-one task)."""
    def __init__(self, repo="fastino/GLiNER2.5-Decide"):
        self.repo = repo
        self.params = 1.0e9 if "1B" in repo else 0.34e9

    # HOW THE MODEL LOADS (wave O; the measurements are in lab/results/opt).
    #   LAB_GLINER_LOAD=direct (the default): the model is built on the device
    #     and the weights are read from the safetensors file in 64 MB pieces
    #     straight into it. No copy of the weights ever sits in RAM, and the
    #     file is not mapped. The same weights, so the same scores, bit for bit.
    #   LAB_GLINER_LOAD=old: the package's own load. It builds the model in
    #     RAM, reads the whole file into RAM next to it, then moves it.
    #   LAB_GLINER_DTYPE=fp16 (the default on the GPU), fp32 (the default on
    #     the CPU) or bf16: the weights' type on the device. fp16 is an
    #     approximation, taken because it keeps the facts: over the 19,920 M9
    #     requests the pick is the same in 99.87% of them, every M9 gate rate
    #     stays inside its 95% interval, and the scores move by 0.005 at most.
    #     It halves the GPU memory and takes a third off the time of a request.
    #     bf16 moved 1% of the picks in 500 requests: not used.
    #   LAB_GLINER_LOAD=old LAB_GLINER_DTYPE=fp32 is the load before wave O.
    DTYPES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}

    # the dtypes of a safetensors file (its header names them)
    ST_DTYPES = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
                 "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
                 "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}

    @staticmethod
    def _copy_safetensors(path, own, chunk=64 << 20):
        """load_state_dict(strict=True) from a safetensors file, read with plain
        reads in pieces of `chunk` bytes straight into `own`'s tensors (wherever
        they live). Neither the file nor a whole tensor is ever mapped or held in
        RAM: the RAM it needs is one piece. Returns None, or why it could not
        (keys, shapes or a dtype that differ: nothing is half-copied then, as the
        checks come first)."""
        import json
        import struct
        with open(path, "rb", buffering=0) as f:
            size = struct.unpack("<Q", f.read(8))[0]
            head = json.loads(f.read(size))
            head.pop("__metadata__", None)
            base = 8 + size
            if set(head) != set(own):
                return "the checkpoint's keys differ from the model's"
            for k, h in head.items():
                if tuple(h["shape"]) != tuple(own[k].shape) or h["dtype"] not in GlinerDecide.ST_DTYPES:
                    return (f"{k} is {h['dtype']} {h['shape']} in the checkpoint,"
                            f" {tuple(own[k].shape)} in the model")
                if not own[k].is_contiguous():
                    return f"{k} is not contiguous in the model"
            buf = bytearray(chunk)
            with torch.no_grad():
                for k, h in head.items():
                    dt = GlinerDecide.ST_DTYPES[h["dtype"]]
                    es = torch.empty(0, dtype=dt).element_size()
                    a, b = h["data_offsets"]
                    flat = own[k].view(-1)
                    f.seek(base + a)
                    step = chunk // es * es
                    for off in range(0, b - a, step):
                        n = min(step, b - a - off)
                        mv = memoryview(buf)[:n]
                        got = f.readinto(mv)
                        if got != n:
                            raise OSError(f"{path}: short read in {k}")
                        flat[off // es:(off + n) // es].copy_(torch.frombuffer(mv, dtype=dt))
        return None

    def _direct(self, device):
        """The model built on `device` with the checkpoint's weights, or None
        when the checkpoint does not match the model exactly (the old load
        then runs, and it says why)."""
        from gliner2.auto import AutoExtractor, _load_config
        from gliner2.configuration import architecture_from_config
        from gliner2.models.base import load_extractor_tokenizer
        from gliner2.models.loading import checkpoint_file
        from transformers import AutoConfig
        # what the package's from_pretrained does, but with the model built
        # on the device (its random start values are thrown away anyway)
        config = _load_config(self.repo, {})
        cls = AutoExtractor._resolve_class(architecture_from_config(config))
        enc = AutoConfig.from_pretrained(checkpoint_file(self.repo, "encoder_config/config.json"))
        with torch.device(device):
            m = cls(config, encoder_config=enc, tokenizer=load_extractor_tokenizer(self.repo),
                    use_flashdeberta=None, word_splitter=None)
        m.config._name_or_path = self.repo
        m.name_or_path = self.repo
        why = self._copy_safetensors(checkpoint_file(self.repo, "model.safetensors"), m.state_dict())
        if why:
            print(f"direct load: {why}; old load", flush=True)
            return None
        return m

    def load(self):
        import gc
        from gliner2 import AutoExtractor
        from gliner2.classification.scoring import ClassificationScorer
        device = os.environ.get("LAB_DEVICE", "cuda")
        how = os.environ.get("LAB_GLINER_LOAD", "direct")
        self.dtype = os.environ.get("LAB_GLINER_DTYPE", "fp16" if device == "cuda" else "fp32")
        self.how = how
        dtype = self.DTYPES[self.dtype]
        m = self._direct(device) if how == "direct" else None
        if m is None:
            m = AutoExtractor.from_pretrained(self.repo)
        adapter = os.environ.get("LAB_GLINER_ADAPTER")
        if adapter:
            # a LoRA trained by lab/methods/finetune_gliner.py
            from gliner2.training.lora import load_lora_adapter
            load_lora_adapter(m, adapter)
        # .to() moves the weights; the constructor's device= only records it
        # (the first full run failed every case: weights on CPU, inputs on GPU).
        # After the direct load it only moves what is not there yet (a LoRA).
        self.scorer = ClassificationScorer(m).to(device=device, dtype=dtype).eval()
        gc.collect()                      # what the load left behind
        if device == "cuda":
            torch.cuda.empty_cache()
        # PRIOR CORRECTION (to-do A4): LAB_PRIOR=<prior_offsets.json> takes each
        # option's favourite-item offset off its log-probability
        # (lab/methods/prior.py; LAB_PRIOR_ALPHA overrides the table's alpha)
        self.prior = None
        if os.environ.get("LAB_PRIOR"):
            import json
            t = json.load(open(os.environ["LAB_PRIOR"]))
            self.prior = (t["offsets"], float(os.environ.get("LAB_PRIOR_ALPHA", t["alpha"])))
        # CAPTURE (runner flag --capture): what the model sees, saved next to
        # the scores. Hooks read the classifier's input (one vector for each
        # option, in label order) and the encoder's output (the text); the
        # package itself is not changed.
        self.capture, self._cap = False, {}

        def on_classifier(mod, args, out):
            if self.capture:
                self._cap["opts"] = args[0].detach().float().cpu()
                self._cap["logits"] = out.detach().float().cpu().reshape(-1)

        def on_encoder(mod, args, kwargs, out):
            if self.capture:
                h = out.last_hidden_state[0].detach().float()
                mask = kwargs.get("attention_mask")
                w = mask[0].float().unsqueeze(-1) if mask is not None else torch.ones_like(h[:, :1])
                self._cap["text_first"] = h[0].cpu()
                self._cap["text_mean"] = ((h * w).sum(0) / w.sum()).cpu()

        m.classifier.register_forward_hook(on_classifier)
        m.encoder.register_forward_hook(on_encoder, with_kwargs=True)

    @staticmethod
    def _ask(question, options):
        """The labels and the schema of one question."""
        from gliner2.classification.schema import ClassificationSchema
        # GLiNER2 refuses a label with a bracket (labels go into its prompt
        # verbatim): "the axe (chop, cut)" is given as "the axe, chop, cut"
        labels = [o.replace(" (", ", ").replace("(", "").replace(")", "") for o in options]
        return labels, ClassificationSchema().single("choice", labels, instruction=question)

    def rank(self, sit, question, options):
        labels, schema = self._ask(question, options)
        self._cap = {}
        sc = self.scorer.score(sit, schema)
        return self._out(sc, labels, options)

    def rank_many(self, asks):
        """rank() of several (situation, question, options) in ONE forward
        pass. The texts are padded to the longest, so the scores can differ
        from rank()'s in the last digits: for offline scoring only
        (score_requests --batch). No capture."""
        if self.capture:
            return [self.rank(*a) for a in asks]
        made = [self._ask(q, o) for _, q, o in asks]
        scs = self.scorer.batch_score([a[0] for a in asks], [m[1] for m in made],
                                      batch_size=len(asks))
        return [self._out(sc, m[0], a[2]) for sc, m, a in zip(scs, made, asks)]

    def _out(self, sc, labels, options):
        p = [sc.probability("choice", o) for o in labels]
        raw = None
        if self.prior:                     # keyed by the option as given, before the bracket fix
            from lab.methods.prior import correct
            raw, p = p, [float(x) for x in correct(p, options, *self.prior)]
        out = {"scores": p, "pick": max(range(len(p)), key=p.__getitem__),
               "order": _order(p), "prob": True}
        if raw is not None:
            out["raw"] = raw
        if self.capture and "opts" in self._cap:
            out["capture"] = dict(self._cap)
        return out


class Bev:
    """avbiswas/bev-decider-0.4B: the first 20 layers of Qwen3-0.6B, a merged
    LoRA and a 2-layer decision head. All options in one pass, but each
    option's tokens see only the question and themselves, so it claims to be
    order-invariant by construction. Licence CC-BY-NC-4.0: testing only."""
    params = 0.45e9

    def load(self):
        from bev_decider import load
        self.d = load(device=os.environ.get("LAB_DEVICE", "cuda"))

    def rank(self, sit, question, options):
        a = self.d.decide(sit, {"q": {"type": "choice", "instructions": question,
                                      "criteria": {o: None for o in options}}})["q"]
        p = [a["probabilities"][o] for o in options]
        return {"scores": p, "pick": max(range(len(p)), key=p.__getitem__),
                "order": _order(p), "prob": True}


class NLI:
    """AlexWortega/openjev: a Qwen3.5 cross-encoder trained for entailment.
    Each option is judged alone ("Kael can use X for this": does it follow?),
    so order only enters the final normalisation."""
    HYP = {"one": "{q} Answer: {o}.",               # the question and the option
           "use": "Kael can use {o} for this.",      # a plain claim
           "work": "{o} would do the job."}          # a claim about the thing

    def __init__(self, sub, method="one"):
        self.sub, self.method = sub, method
        self.params = 2.0e9 if "2b" in sub else 0.8e9

    def load(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("AlexWortega/openjev", subfolder=self.sub)
        self.m = AutoModelForSequenceClassification.from_pretrained(
            "AlexWortega/openjev", subfolder=self.sub, dtype=torch.bfloat16).cuda().eval()
        labels = {v.lower(): int(k) for k, v in self.m.config.id2label.items()}
        self.ent = labels.get("entailment", 1)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.m.config.pad_token_id = self.tok.pad_token_id

    @torch.no_grad()
    def rank(self, sit, question, options, bs=16):
        sc = []
        for i in range(0, len(options), bs):
            hyp = [self.HYP[self.method].format(q=question, o=o) for o in options[i:i + bs]]
            b = self.tok([sit] * len(hyp), hyp, return_tensors="pt", padding=True,
                         truncation=True, max_length=1024).to("cuda")
            lp = torch.log_softmax(self.m(**b).logits.float(), -1)[:, self.ent]
            sc += [float(x) for x in lp.cpu()]
        return {"scores": sc, "pick": max(range(len(sc)), key=sc.__getitem__),
                "order": _order(sc), "prob": False}


class PermAverage:
    """Ask in k orders and average the probabilities, reported in the original
    order (the pijev / permutation self-consistency idea): original, reversed,
    then cyclic shifts."""
    def __init__(self, base, k):
        self.base, self.k = base, k
        self.params = base.params

    def load(self):
        self.base.load()

    def rank(self, sit, question, options):
        n = len(options)
        orders = [list(range(n)), list(range(n))[::-1]]
        step = max(1, n // max(self.k - 1, 1))
        while len(orders) < self.k:
            s = step * (len(orders) - 1)
            orders.append([(i + s) % n for i in range(n)])
        acc = [0.0] * n
        for od in orders[:self.k]:
            r = self.base.rank(sit, question, [options[i] for i in od])
            p = r["scores"] if abs(sum(r["scores"]) - 1) < 1e-3 else _softmax(r["scores"])
            for j, i in enumerate(od):
                acc[i] += p[j] / self.k
        return {"scores": acc, "pick": max(range(n), key=acc.__getitem__),
                "order": _order(acc), "prob": True}


class Tournament:
    """The user's idea (2026-09-28): split a large pool into groups of the size
    where the model is still good, keep the best of each group, and group the
    winners again, until one is left. The same shape as TourRank's group
    stages and the "setwise" rankers.

        g        group size (the model reads g options at a time)
        keep     how many advance from each group (2 forgives a near miss)
        repeats  independent random groupings; each finalist earns points,
                 which also averages out position bias and unlucky draws

    Each option also gets a score for drawing: the product of the
    probabilities it won with, along its path (a hierarchical softmax)."""
    def __init__(self, base, g=4, keep=1, repeats=1):
        self.base, self.g, self.keep, self.repeats = base, g, keep, repeats
        self.params = base.params

    def load(self):
        self.base.load()

    def _groups_scores(self, sit, question, options, groups):
        # Laya reads many questions in one call: a whole round in a few calls
        if isinstance(self.base, Laya) and self.base.method == "all":
            qs = {f"g{j}": {"type": "choice", "instructions": question,
                            "criteria": {options[i]: None for i in grp}}
                  for j, grp in enumerate(groups)}
            ans = self.base._predict(sit, qs)
            return [[ans[f"g{j}"]["probabilities"][options[i]] for i in grp]
                    for j, grp in enumerate(groups)]
        out = []
        for grp in groups:
            r = self.base.rank(sit, question, [options[i] for i in grp])
            sc = r["scores"]
            if sc is None:          # a knockout base: turn its order into weights
                sc = [0.0] * len(grp)
                for rank_, i in enumerate(r["order"]):
                    sc[i] = 1.0 / (rank_ + 1)
            p = sc if abs(sum(sc) - 1) < 1e-3 else _softmax(sc)
            out.append(p)
        return out

    def rank(self, sit, question, options):
        import random as _r
        n = len(options)
        rng = _r.Random(hash(tuple(options)) & 0xffffffff)
        points, path = [0.0] * n, [0.0] * n
        for _ in range(self.repeats):
            pool = list(range(n))
            rng.shuffle(pool)
            logp = {i: 0.0 for i in pool}
            while len(pool) > 1:
                if len(pool) <= self.g:
                    groups, keep = [pool], 1
                else:
                    groups = [pool[k:k + self.g] for k in range(0, len(pool), self.g)]
                    keep = self.keep
                probs = self._groups_scores(sit, question, options, groups)
                nxt = []
                for grp, p in zip(groups, probs):
                    for i, pi in zip(grp, p):
                        logp[i] += math.log(max(pi, 1e-9))
                    best = sorted(range(len(grp)), key=lambda k: -p[k])[:keep if len(grp) > 1 else 1]
                    nxt += [grp[k] for k in best]
                rng.shuffle(nxt)
                if len(nxt) == len(pool):      # nothing was cut: stop to avoid looping
                    break
                pool = nxt
            points[pool[0]] += 1
            for i, lp in logp.items():
                path[i] += lp / self.repeats
        order = sorted(range(n), key=lambda i: (-points[i], -path[i]))
        return {"scores": path, "pick": order[0], "order": order, "prob": False}


class Overlap:
    """No model: the number of content words an option shares with the
    situation and question. The leak check for every presentation."""
    params = 0

    def load(self):
        pass

    def rank(self, sit, question, options):
        import hashlib
        from lab.methods.failures import words
        ctx = words(f"{sit} {question}")
        # ties broken by a fixed hash of the text, not by position
        sc = [len(words(o) & ctx) + int(hashlib.md5(o.encode()).hexdigest()[:6], 16) / 1e9 for o in options]
        return {"scores": sc, "pick": max(range(len(sc)), key=sc.__getitem__),
                "order": _order(sc), "prob": False}


class Hier:
    """TWO STEPS, as the grouped pick-list: choose the kind of thing first
    (tools, kitchen things, ...), then the thing within that kind. Each kind is
    shown with up to three of its things. Things cases only (the kinds come
    from the item table)."""
    def __init__(self, inner):
        self.inner, self.params = inner, inner.params

    def load(self):
        from lab.methods.cases import parse
        from lab.methods.present import KIND_WORDS
        self.kind = {i["name"]: i["cat"] for i in parse()[0]}
        self.words = KIND_WORDS
        self.inner.load()

    def rank(self, sit, question, options):
        groups = {}
        for i, o in enumerate(options):
            groups.setdefault(self.kind[o.split(":")[0]], []).append(i)
        kinds = list(groups)
        if len(kinds) == 1:
            return self.inner.rank(sit, question, options)
        labels = [f"{self.words[k]} ({', '.join(options[i] for i in groups[k][:3])}"
                  f"{', and more' if len(groups[k]) > 3 else ''})" for k in kinds]
        q1 = question.replace("Which thing", "Which kind of thing")
        r1 = self.inner.rank(sit, q1, labels)
        g = groups[kinds[r1["pick"]]]
        r2 = self.inner.rank(sit, question, [options[i] for i in g])
        order = [g[j] for j in r2["order"]]
        order += [i for k in r1["order"] for i in groups[kinds[k]] if i not in order]
        return {"scores": None, "pick": g[r2["pick"]], "order": order, "prob": False}


def make(name):
    model, method = name.split(":")
    if method.startswith("perm"):                 # perm4 = ask in 4 orders, average
        return PermAverage(make(f"{model}:all"), int(method[4:]))
    if model == "bev" and method == "all":
        return Bev()
    if model == "von13":                           # runs in lab/vendor/.venv-von13
        return Von(method)
    if model == "overlap":
        return Overlap()
    if method == "hier":
        return Hier(make(f"{model}:all" if model not in ("onli2b", "onli08b") else f"{model}:one"))
    if model == "onli2b" and method in NLI.HYP:
        return NLI("qwen3.5-2b-nli-v5", method)
    if model == "onli08b" and method in NLI.HYP:
        return NLI("qwen3.5-0.8b-nli-v2s-long", method)
    if method.startswith("tour"):
        # tour4 = groups of 4, keep 1; tour8k2 = groups of 8, keep 2; tour4x3 = 3 groupings
        import re as _re
        m_ = _re.fullmatch(r"tour(\d+)(?:k(\d+))?(?:x(\d+))?", method)
        g, keep, reps = int(m_.group(1)), int(m_.group(2) or 1), int(m_.group(3) or 1)
        return Tournament(make(f"{model}:all"), g=g, keep=keep, repeats=reps)
    if model == "laya_en":
        return Laya(method)
    if model == "laya_multi":
        s = Laya(method, sub="multilingual")
        s.params = 0.35e9
        return s
    if model == "von":
        return Von(method)
    if model == "gdecide" and method == "all":
        return GlinerDecide()
    if model == "gdecide1b" and method == "all":
        return GlinerDecide("fastino/GLiNER2.5-Decide-1B")
    if model == "qwen06":
        return Qwen("Qwen/Qwen3-0.6B", method)
    if model == "qwen17":
        return Qwen("Qwen/Qwen3-1.7B", method)
    if model == "bge" and method == "apart":
        return Embed()
    if model == "bge_qwen17" and method == "short":
        return Shortlist(Embed(), Qwen("Qwen/Qwen3-1.7B", "one"))
    raise ValueError(name)
