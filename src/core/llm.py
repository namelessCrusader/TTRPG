"""Shared local-LM utility: one cached model per id, and the PMI choice primitive.

Everything the engine asks of a small model is SELECTION — pick one option id
from a short list. `ChoiceLM.pick` scores each option's tokens under the real
context vs a neutral context (contrastive PMI) so the model's prior toward a
favourite word cancels out, and picks the argmax. Single forward pass per
option, no generation, no parsing — a 0.6B does this reliably at ~50-150ms.
"""

from __future__ import annotations

_CACHE: dict = {}


def assemble(slots: list, budget: int) -> str:
    """Harness #3: prompts are ASSEMBLED, not concatenated. Slots are
    (name, text, tier); render in given order, but under budget drop WHOLE slots
    lowest-tier-first (highest tier number = most expendable). Deterministic and
    byte-stable, so prefix caches and PMI calibration hold across turns."""
    keep = list(range(len(slots)))
    def total(ix): return sum(len(slots[i][1]) + 1 for i in ix)
    while keep and total(keep) > budget:
        worst = max(keep, key=lambda i: (slots[i][2], i))
        keep.remove(worst)
    return "\n".join(slots[i][1] for i in keep)


def get_lm(model_id: str = "Qwen/Qwen3-0.6B") -> "ChoiceLM":
    if model_id not in _CACHE:
        _CACHE[model_id] = ChoiceLM(model_id)
    return _CACHE[model_id]


class ChoiceLM:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._tok = self._model = self._dev = None
        self._priors = {}    # (neutral_prompt, option) → cached PMI denominator

    def load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._dev = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self._dev == "cuda" else torch.float32
            self._tok = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=dtype, low_cpu_mem_usage=True).to(self._dev).eval()
        return self

    def _prompt(self, sysmsg: str, body: str) -> str:
        msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": body}]
        try:   # Qwen3 thinking toggle; we only score, so disable it
            return self._tok.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _avg_logp(self, prompt: str, cont: list) -> float:
        import torch
        pids = self._tok(prompt, add_special_tokens=False).input_ids
        inp = torch.tensor([pids + cont], device=self._dev)
        with torch.no_grad():
            logits = self._model(inp).logits[0].float()
        lsm = torch.nn.functional.log_softmax
        return sum(lsm(logits[len(pids) + j - 1], dim=-1)[tid].item()
                   for j, tid in enumerate(cont)) / max(1, len(cont))

    CHUNK = 4    # rows per forward: full-vocab logits are the memory hog on 6GB

    def _batch_logps(self, rows):
        """Padded micro-batched forwards for (prompt_ids, cont_ids) rows → avg
        logps. Gathers only continuation positions (never casts the full logits
        tensor) so a whole menu scores in a handful of forwards, OOM-free."""
        import torch
        lsm = torch.nn.functional.log_softmax
        pad = self._tok.pad_token_id or self._tok.eos_token_id
        out = []
        for k in range(0, len(rows), self.CHUNK):
            chunk = rows[k:k + self.CHUNK]
            maxlen = max(len(p) + len(c) for p, c in chunk)
            inp = torch.full((len(chunk), maxlen), pad, dtype=torch.long)
            mask = torch.zeros((len(chunk), maxlen), dtype=torch.long)
            for i, (p, c) in enumerate(chunk):
                n = len(p) + len(c)
                inp[i, :n] = torch.tensor(p + c)
                mask[i, :n] = 1
            with torch.no_grad():
                logits = self._model(inp.to(self._dev),
                                     attention_mask=mask.to(self._dev)).logits
            for i, (p, c) in enumerate(chunk):
                lp = sum(lsm(logits[i, len(p) + j - 1].float(), dim=-1)[tid].item()
                         for j, tid in enumerate(c))
                out.append(lp / max(1, len(c)))
            del logits
        return out

    def pick_scored(self, sysmsg: str, cond_body: str, neutral_body: str, options: list):
        """PMI per option: logP(opt | cond) − logP(opt | neutral). Numerators batch
        into one forward; denominators are CACHED (the neutral prompt recurs every
        turn per call site, so after warmup PMI costs numerators only)."""
        self.load()
        cond = self._tok(self._prompt(sysmsg, cond_body), add_special_tokens=False).input_ids
        neut_p = self._prompt(sysmsg, neutral_body)
        neut = self._tok(neut_p, add_special_tokens=False).input_ids
        conts = [self._tok(str(o), add_special_tokens=False).input_ids for o in options]
        rows, need = [(cond, c) for c in conts], []
        for o, c in zip(options, conts):
            if (neut_p, str(o)) not in self._priors:
                rows.append((neut, c))
                need.append((neut_p, str(o)))
        lps = self._batch_logps(rows)
        for k, lp in zip(need, lps[len(conts):]):
            self._priors[k] = lp
        scores = [lps[i] - self._priors[(neut_p, str(o))] for i, o in enumerate(options)]
        return max(range(len(options)), key=lambda i: scores[i]), scores

    def pick(self, sysmsg: str, cond_body: str, neutral_body: str, options: list) -> int:
        return self.pick_scored(sysmsg, cond_body, neutral_body, options)[0]
