#!/usr/bin/env python3
"""
Run short autonomous sims on random worlds and report human-likeness issues.

Usage:
  python3 scripts/auto_smoke_audit.py [--rounds 5] [--ticks 3] [--ollama]
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WORLDS = ["tavern", "magic_duel", "castle", "spicy", "default"]

# Patterns that indicate non-human NPC output
ISSUE_PATTERNS = [
    ("observe_spam", re.compile(r"action=observe", re.I)),
    ("wait_spam", re.compile(r"action=wait", re.I)),
    ("flee_spam", re.compile(r"action=flee", re.I)),
    ("planning_speech", re.compile(
        r"(ask if|should I|I need to|let me think|as an AI|option \d)", re.I
    )),
    ("goal_as_dialogue", re.compile(
        r"says.*without getting involved|says.*decide whether to|says.*get through tonight",
        re.I,
    )),
    ("impossible_temp", re.compile(r"now -?\d{4,}°C")),
    ("empty_cast", re.compile(r"no visible effect", re.I)),
    ("raw_bracket", re.compile(r"\[npc_auto:", re.I)),
    ("duplicate_line", None),  # handled separately
]


def _capture_run(
    world_name: str,
    ticks: int,
    *,
    use_ollama: bool,
    use_torch: bool,
    torch_path: str | None,
    model: str,
    cognition: str | None,
) -> tuple[str, list[str]]:
    from contextlib import redirect_stdout

    from src.sim.autonomous import run_simulation

    world_path = ROOT / "worlds" / world_name
    if not world_path.is_dir():
        return "", [f"world_missing:{world_name}"]

    buf = StringIO()
    issues: list[str] = []
    try:
        with redirect_stdout(buf):
            run_simulation(
                world_path,
                ticks=ticks,
                map_every=0,
                use_ollama=use_ollama,
                use_torch=use_torch,
                torch_path=torch_path,
                use_mock=not (use_ollama or use_torch),
                model=model,
                quiet=False,
                prerun_history_ticks=0 if world_name == "magic_duel" else 3,
                cognition_mode=cognition,
            )
    except Exception as exc:
        issues.append(f"crash:{exc}")
    return buf.getvalue(), issues


def _analyze_output(text: str, world: str) -> list[str]:
    issues: list[str] = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    actions = re.findall(
        r"\] tick=\d+ action=(\w+)(?! rejected)", text, re.I
    )
    ac = Counter(a.lower() for a in actions)
    n_actions = sum(ac.values()) or 1

    if ac.get("observe", 0) / n_actions > 0.4:
        issues.append(f"observe_spam:{ac['observe']}/{n_actions}")
    if ac.get("wait", 0) / n_actions > 0.35:
        issues.append(f"wait_spam:{ac['wait']}/{n_actions}")
    if ac.get("flee", 0) / n_actions > 0.25:
        issues.append(f"flee_spam:{ac['flee']}/{n_actions}")

    narr = [ln for ln in lines if ln.startswith("    ") and "action=" not in ln]
    if len(narr) >= 2 and len(set(narr)) < len(narr) * 0.6:
        issues.append("repetitive_narration")

    for name, pat in ISSUE_PATTERNS:
        if pat is None:
            continue
        if pat.search(text):
            issues.append(name)

    if world == "magic_duel":
        if ac.get("cast", 0) < max(1, n_actions // 3):
            issues.append(f"duel_low_casts:{ac.get('cast', 0)}/{n_actions}")
        if "PHYSICS" not in text:
            issues.append("duel_no_physics_lines")

    if world == "tavern":
        quoted = len(re.findall(r'says[,:]?\s*["\u201c]', text, re.I))
        quoted += len(re.findall(r'\bspeaks\b.*["\u201c]', text, re.I))
        if ac.get("speak", 0) == 0 and quoted < 2:
            issues.append("tavern_no_social_actions")
        if ac.get("speak", 0) / n_actions > 0.72:
            issues.append(f"speak_spam:{ac.get('speak', 0)}/{n_actions}")
        if ac.get("observe", 0) > ac.get("speak", 0) and ac.get("speak", 0) > 0:
            issues.append("tavern_more_observe_than_speak")

    return issues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--ticks", type=int, default=3)
    ap.add_argument("--ollama", action="store_true")
    ap.add_argument("--torch", action="store_true")
    ap.add_argument("--torch-path", dest="torch_path", default=None, metavar="DIR")
    ap.add_argument("--model", default="qwen3:1.7b")
    ap.add_argument("--cognition", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    all_issues: list[tuple[str, list[str]]] = []

    for i in range(args.rounds):
        world = rng.choice(WORLDS)
        print(f"\n{'='*60}\nROUND {i+1}/{args.rounds}: {world} ({args.ticks} ticks)\n{'='*60}")
        text, pre = _capture_run(
            world,
            args.ticks,
            use_ollama=args.ollama,
            use_torch=args.torch,
            torch_path=args.torch_path,
            model=args.model,
            cognition=args.cognition,
        )
        found = pre + _analyze_output(text, world)
        all_issues.append((world, found))
        if found:
            print("ISSUES:", ", ".join(found))
        else:
            print("ISSUES: none detected")
        # Show last ~25 lines of sim output
        tail = "\n".join(text.splitlines()[-25:])
        print(tail)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    by_issue: Counter[str] = Counter()
    for world, issues in all_issues:
        for iss in issues:
            by_issue[iss.split(":")[0]] += 1
        print(f"  {world}: {issues or ['ok']}")
    print("\nTop issue types:", dict(by_issue.most_common(10)))
    return 1 if by_issue else 0


if __name__ == "__main__":
    raise SystemExit(main())
