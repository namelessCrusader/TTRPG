"""A LoRA fine-tune of GLiNER2.5-Decide (plan v4, 5d). Prepared on the laptop;
the training runs on the other machine.

    python3 -m lab.methods.finetune_gliner data          # lab/methods/ft/*.jsonl (CPU, seconds)
    python  -m lab.methods.finetune_gliner train --out lab/checkpoints/gliner_ft [--epochs 3 --r 8]
    # then score the held-out lists with and without the adapter:
    lab/.venv/bin/python -m lab.methods.run gdecide:all --cases lab/methods/ft/cases_test.jsonl --tag ft_base
    LAB_GLINER_ADAPTER=<the saved adapter dir> \\
      lab/.venv/bin/python -m lab.methods.run gdecide:all --cases lab/methods/ft/cases_test.jsonl --tag ft

WHAT IT LEARNS. The things and creativity lists as GLiNER2 classification
examples: text = the situation, prompt = the question, labels = the options,
true label = the acceptable option. Only lists with exactly one acceptable
option per example: a list with several becomes one example per acceptable
option, with the others taken out (a pick-one task; several true labels
would turn it into a many-label task with another loss). Hard negatives come
from the lists themselves: the hand-picked near misses and the
unusual-but-workable cells.

HELD OUT, so the test says whether it LEARNED rather than remembered:
    TEST_NEEDS   8 needs that never appear in training
    TEST_ITEMS   15 things taken out of every training list
cases_test.jsonl holds every list with a test need, or with a test thing as
the acceptable answer.

NOT YET: the entropy and "no need gives an even output" terms of plan
section 3 need a custom training loop; the package trainer takes a plain
classification loss. Run this first, then add them if the favourites remain.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
FT = HERE / "ft"
TEST_NEEDS = ["cut the rope", "light the dark cellar", "fix the loose nail in the door", "pay the ferryman",
              "cross the flooded stream", "mend the torn cloak", "trap a rabbit", "pour the oil into the lamp"]
TEST_ITEMS = ["the kitchen knife", "the lantern", "the hammer", "the silver coin", "the plank", "the needle and thread",
              "the snare", "the funnel", "the scythe", "the tinderbox", "the bucket", "the shovel", "the ladder",
              "the blanket", "the horn"]


def data(seed=0):
    rng = random.Random(seed)
    FT.mkdir(exist_ok=True)
    src = []
    for f in ("cases_things.jsonl", "cases_creative.jsonl"):
        src += [json.loads(l) for l in open(HERE / f)]
    train, test = [], []
    for c in src:
        answers = {c["names"][i] for i in c["good"]}
        if c["need"] in TEST_NEEDS or answers & set(TEST_ITEMS):
            test.append(c)                   # a new need, or a new thing as the answer
            continue
        # the held-out things never appear in training: they are taken out of
        # the list. A list with several acceptable options becomes one
        # example per option, with the other acceptable ones taken out, so
        # it stays a pick-one task.
        keep = [i for i, n in enumerate(c["names"]) if n not in TEST_ITEMS]
        for g in c["good"]:
            idx = [i for i in keep if i == g or i not in c["good"]]
            train.append({"text": c["situation"], "prompt": c["question"],
                          "labels": [c["options"][i] for i in idx], "true": c["options"][g],
                          "id": f"{c['id']}#{g}", "need": c["need"]})
    rng.shuffle(train)
    k = max(1, len(train) // 10)
    ev, tr = train[:k], train[k:]
    for name, rows in (("train", tr), ("eval", ev)):
        with open(FT / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    with open(FT / "cases_test.jsonl", "w") as f:
        for c in test:
            f.write(json.dumps(c) + "\n")
    print(f"train {len(tr)}, eval {len(ev)}, held-out lists {len(test)} "
          f"({sum(c['need'] in TEST_NEEDS for c in test)} with a new need) -> {FT}")


def train(out, epochs=3, r=8, lr=1e-5, batch=4):
    from gliner2 import AutoExtractor
    from gliner2.training.data import Classification, InputExample
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

    def examples(name):
        rows = [json.loads(l) for l in open(FT / f"{name}.jsonl")]
        return [InputExample(text=x["text"], classifications=[Classification(
            task="choice", labels=x["labels"], true_label=x["true"], prompt=x["prompt"])]) for x in rows]

    model = AutoExtractor.from_pretrained("fastino/GLiNER2.5-Decide")
    cfg = TrainingConfig(output_dir=out, experiment_name="gliner_decide_ft", num_epochs=epochs, batch_size=batch,
                         encoder_lr=lr, use_lora=True, lora_r=r, lora_alpha=2 * r, save_adapter_only=True,
                         bf16=True, eval_steps=200, save_best=True, num_workers=2)
    GLiNER2Trainer(model, cfg).train(train_data=examples("train"), eval_data=examples("eval"))
    print(f"saved under {out}; point LAB_GLINER_ADAPTER at the adapter folder there")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "train"])
    ap.add_argument("--out", default="lab/checkpoints/gliner_ft")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--r", type=int, default=8)
    a = ap.parse_args()
    data() if a.cmd == "data" else train(a.out, a.epochs, a.r)
