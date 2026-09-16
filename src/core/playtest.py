"""playtest.py — the play-test loop on the real engine (the decision-2 loop).

One beat: the protagonist gets its VIEW (what it perceives) and its PIECE TABLE (the legal
pieces). An author — a person or an AI — answers with one composed line or tree. The parser
checks every piece; the engine runs the plan; the other characters act on their own brains
(mind.py) inside the same turn. Everything is logged as corpus rows.

A refusal costs no time: the author is told the exact bad piece and may answer again.
CLI (state kept in a pickle):
  python -m src.core.playtest new <scene> <folder> [seed]
  python -m src.core.playtest show <folder>
  python -m src.core.playtest act <folder> "<composed answer>"
"""
import json
import os
import pickle
import sys

from . import engine, perceive, pieces

BRIEFS = {
    "vault":     "Steal the starfire vial from the vault and get out. Bran guards it; "
                 "Sly wants it too. You carry a torch, oil, acid and water.",
    "guildhall": "Auction night. The vial sits in the vault room; the storeroom is locked. "
                 "Get the vial by wit, fire, or acid — and get out.",
    "dungeon":   "Loot the floor and reach the stairs down. The scuttler and the "
                 "broodmother hunt on their own.",
    "farm":      "Tend the farm; deal with the proud rooster; earn Meiling's trust.",
}


PACKS = {"dungeon": ["dcc"]}          # scenes that come with a system pack on


def new(scene, seed=0):
    from . import reactions
    from . import seed as seeds
    world, player = getattr(seeds, scene)(seed=seed)
    world.facts["scene"] = scene
    for pk in PACKS.get(scene, []):   # run 2 found the dungeon shipped WITHOUT its dcc
        world.packs.append(reactions.load_pack(pk))   # pack — no open, no descend, no xp
    return world, player


def prompt(world, player) -> str:
    """Everything the author reads for one beat: goal, view, table, how to answer."""
    perceive.observe(world, player, world.tick)
    slice_ = perceive.perceive(world, player, now=world.tick)
    view = perceive.render(slice_)
    tbl = pieces.table(engine.affordance_menu(world, player, slice_=slice_), slice_=slice_)
    return (f"YOUR GOAL: {BRIEFS.get(world.facts.get('scene'), '(free play)')}\n\n"
            f"{view}\n\n{tbl}\n\n"
            "Answer with ONE composed line or tree, from listed pieces only. Examples:\n"
            "  pour water at:SE\n"
            "  (then (take target:brazier) (move dir:E))\n"
            "  (if (near guard) (move dir:W) (smash target:cask_water))\n"
            "For speak, put your words in quotes: speak to:guard \"Stand aside.\" — quotes keep\n"
            "the words inside their node, so speak works inside (then …) trees too.")


def act(world, player, text) -> dict:
    """Parse and run one answer. Returns what happened — or the named bad piece."""
    if "|" in text and not text.lstrip().startswith("("):     # flat-line comfort form:
        text, _, w = text.partition("|")                      # speak to:x | words
        text = f'{text.strip()} "{w.strip()}"'
    perceive.observe(world, player, world.tick)
    slice_ = perceive.perceive(world, player, now=world.tick)
    opts = engine.affordance_menu(world, player, slice_=slice_)
    try:
        plan = pieces.parse(text, opts, slice_=slice_)
    except pieces.PieceError as e:
        return {"ok": False, "refused": str(e), "piece": e.piece, "narrative": []}
    lines, ran = [], []
    for cond, opt, said in plan:
        if not pieces.check(world, player, cond):
            lines.append(f"({opt.verb}: its condition is not true — skipped)")
            continue
        # re-resolve the piece against the FRESH world: a later step in a plan must act
        # from where the world now stands, not from a stale copy (e.g. the second
        # "move dir:S" walks from the NEW position; a vanished target skips by name).
        key = (opt.verb, tuple(sorted(opt.args.items())))
        live = pieces.index(engine.affordance_menu(world, player)).get(key)
        if live is None:
            lines.append(f"({opt.verb}: no longer possible — skipped)")
            continue
        if live.social and said:
            live.social = (live.social[0], said)
        out = engine.step(world, live, actor=player)
        lines += out["narrative"]
        ran.append(live.verb)
    return {"ok": True, "ran": ran, "narrative": lines,
            "over": engine.is_over(world), "violations": engine.invariants(world)}


# ── CLI (pickled state per folder, plus a corpus of (prompt → answer → result)) ──
def _p(folder):
    return os.path.join(folder, "playtest.pkl"), os.path.join(folder, "corpus.jsonl")


def main(argv):
    cmd, folder = argv[0], argv[1] if len(argv) > 1 else "."
    pkl, corpus = _p(folder)
    if cmd == "new":
        scene, seed = argv[1], int(argv[3]) if len(argv) > 3 else 0
        pkl, corpus = _p(argv[2])
        os.makedirs(argv[2], exist_ok=True)
        world, player = new(scene, seed)
        pickle.dump((world, player.id), open(pkl, "wb"))
        print(prompt(world, world.entities[player.id]))
    elif cmd == "show":
        world, pid = pickle.load(open(pkl, "rb"))
        print(prompt(world, world.entities[pid]))
    elif cmd == "act":
        world, pid = pickle.load(open(pkl, "rb"))
        player = world.entities[pid]
        text = argv[2]
        row = {"tick": world.tick, "prompt": prompt(world, player), "answer": text}
        out = act(world, player, text)
        row["result"] = {k: out[k] for k in out if k != "narrative"} | {"story": out["narrative"]}
        with open(corpus, "a") as f:
            f.write(json.dumps(row) + "\n")
        pickle.dump((world, pid), open(pkl, "wb"))
        for ln in out["narrative"] or [out.get("refused", "")]:
            print(ln)
        if not out["ok"]:
            print(f">> refused — bad piece: {out['piece']!r}. No time passed. Answer again.")
        elif out["over"]:
            print(">> the story is over.")
        else:
            print("\n" + prompt(world, player))


if __name__ == "__main__":
    main(sys.argv[1:])
