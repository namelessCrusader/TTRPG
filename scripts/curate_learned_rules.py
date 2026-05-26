#!/usr/bin/env python3
"""
Curate learned adjudication rules from ``<pack>/learned/*.yaml``.

The engine appends candidate verb templates when ``learn_from_adjudication: true``.
This script lists staging entries and optionally promotes them into the canonical
``verb_templates.yaml`` (human review still recommended).

Usage:
    python scripts/curate_learned_rules.py worlds/tavern
    python scripts/curate_learned_rules.py worlds/tavern --promote --dry-run
    python scripts/curate_learned_rules.py worlds/tavern --promote
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _list_learned(pack: Path) -> list[tuple[Path, str]]:
    learned_dir = pack / "learned"
    if not learned_dir.is_dir():
        return []
    out: list[tuple[Path, str]] = []
    for path in sorted(learned_dir.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        verbs = re.findall(r"^\s*-\s*verb:\s*(\S+)", text, re.MULTILINE)
        header = f"{path.name} ({len(verbs)} verb(s): {', '.join(verbs[:6])})"
        out.append((path, header))
    return out


def _extract_template_blocks(text: str) -> list[str]:
    """Return YAML list items starting with ``- verb:``."""
    blocks: list[str] = []
    current: list[str] = []
    in_block = False
    for line in text.splitlines():
        if re.match(r"^\s*-\s*verb:\s*", line):
            if current:
                blocks.append("\n".join(current).rstrip())
            current = [line]
            in_block = True
        elif in_block:
            if line.startswith("#") or line.strip() == "" or line.startswith(" "):
                current.append(line)
            else:
                blocks.append("\n".join(current).rstrip())
                current = []
                in_block = False
    if current:
        blocks.append("\n".join(current).rstrip())
    return blocks


def _verb_in_canonical(canonical: Path, verb: str) -> bool:
    if not canonical.is_file():
        return False
    text = canonical.read_text(encoding="utf-8")
    return bool(re.search(rf"^\s*-\s*verb:\s*{re.escape(verb)}\s*$", text, re.MULTILINE))


def promote(pack: Path, *, dry_run: bool) -> int:
    learned_dir = pack / "learned"
    canonical = pack / "verb_templates.yaml"
    if not learned_dir.is_dir():
        print(f"No learned/ folder in {pack}")
        return 0

    promoted = 0
    append_lines: list[str] = []
    for path in sorted(learned_dir.glob("*.yaml")):
        for block in _extract_template_blocks(path.read_text(encoding="utf-8")):
            m = re.search(r"verb:\s*(\S+)", block)
            if not m:
                continue
            verb = m.group(1)
            if _verb_in_canonical(canonical, verb):
                print(f"  skip {verb} (already in verb_templates.yaml)")
                continue
            promoted += 1
            append_lines.append(f"\n# promoted from {path.name}\n{block}\n")
            print(f"  promote {verb} from {path.name}")

    if promoted == 0:
        print("Nothing to promote.")
        return 0

    if dry_run:
        print(f"\n[dry-run] Would append {promoted} template(s) to {canonical}")
        return promoted

    canonical.parent.mkdir(parents=True, exist_ok=True)
    if not canonical.is_file():
        canonical.write_text("templates: []\n", encoding="utf-8")
    with canonical.open("a", encoding="utf-8") as fh:
        fh.write("\n# ── promoted from learned/ ──\n")
        for chunk in append_lines:
            fh.write(chunk)
    print(f"\nAppended {promoted} template(s) to {canonical}")
    return promoted


def main() -> int:
    parser = argparse.ArgumentParser(description="Curate learned adjudication rules")
    parser.add_argument("pack", help="World pack directory (e.g. worlds/tavern)")
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Append new learned verbs to verb_templates.yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --promote, print actions without writing",
    )
    args = parser.parse_args()

    pack = Path(args.pack)
    if not pack.is_absolute():
        pack = _repo_root() / pack
    if not pack.is_dir():
        print(f"Pack not found: {pack}", file=sys.stderr)
        return 1

    entries = _list_learned(pack)
    if not entries:
        print(f"No learned rules in {pack / 'learned'}")
        print("Enable with world.yaml: extra.learn_from_adjudication: true")
        return 0

    print(f"Learned staging in {pack / 'learned'}:")
    for _, header in entries:
        print(f"  • {header}")

    if args.promote:
        promote(pack, dry_run=args.dry_run)
    else:
        print("\nRun with --promote --dry-run to preview promotion to verb_templates.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
