# The mind lab

Tests the mental and social half (deciding, reading people, speaking) in
isolation from the simulation, so the two can move in parallel.

**The isolation rule.** Nothing in `lab/` imports `src/`, and nothing in `src/`
imports `lab/`. The only thing that crosses is a card file (`lab/cards/*.jsonl`):
a state as text, a set of typed questions, and the gold answers. The lab has its
own venv, so torch and the model packages never touch the simulation's
environment. When the sim can write social situations, it writes cards in the
same format and the lab scores models on them unchanged.

```
uv venv lab/.venv --python 3.12
uv pip install --python lab/.venv -r lab/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match

python3 lab/cards/social_v1.py                          # rebuild the cards
lab/.venv/bin/python -m lab.run majority                # the line to clear
lab/.venv/bin/python -m lab.run laya                    # von, gliclass, qwen
lab/.venv/bin/python -m lab.speak --voice qwen --reader laya
```

Run one at a time. The laptop has 14 GB of RAM and 6 GB of VRAM, and the
simulation's test suites want most of the RAM.

## What is tested

| file | question |
| --- | --- |
| `run.py` | READ: what is a line doing, to whom, how hostile, does it ask for something? MIND: what does this person do, how afraid are they, do they believe it? |
| `speak.py` | Given a decision, can a small LM say it so a reader recovers the act (HEARD), without inventing names the scene never held (GROUNDED, checked by GLiNER)? |

The MIND gold answers are hand-labelled, a reasonable GM's call. They are
provisional until the sim can produce the situations and score their outcomes.

## The models

| name | what | licence |
| --- | --- | --- |
| `laya` | convaiinnovations/laya: ModernBERT-large with a typed-decision head | Apache-2.0 |
| `von` | wfzyx/von-1.0: ModernBERT-large, System One decisions | Apache-2.0 |
| `gliclass` | knowledgator GLiClass: labels ride in the input | Apache-2.0 |
| `qwen` | Qwen3-0.6B as a picker, by mean log-probability per option | Apache-2.0 |
| GLiNER | urchade/gliner_medium-v2.1, the entity check in `speak.py` | Apache-2.0 |

Not here yet: **jevlike** ships an architecture and trainer but no general
checkpoint, so it needs our data first. **GLiFormer** has no public code or
weights. **Jev** is closed.
