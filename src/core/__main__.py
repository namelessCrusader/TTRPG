"""Drivers: human play, a free random-bot coherence loop, and a step-server
interface so *any* agent (a cheap LM, over repeated `step` calls) can play.

  python -m src.core new                 # start a game, print scene + menu
  python -m src.core step 3              # take menu option 3, print result
  python -m src.core auto --turns 2000   # random bot hammers coherence, reports
  python -m src.core play                # interactive human loop
"""

from __future__ import annotations

import argparse
import pickle
import random
import sys

from . import engine
from .seed import guildhall as scene

_SCRATCH = "/tmp/claude-1000/-home-nisargparikh-Desktop-Fun-Stuff-Mark-1/88440bcd-3d95-4f09-8a13-ebe021f962f7/scratchpad"
STATE = _SCRATCH + "/mark1_core.pkl"
# playtest session: world + the harvested corpus + notes, persisted between CLI calls
SESS = _SCRATCH + "/playtest_world.pkl"
CORPUS = _SCRATCH + "/playtest_corpus.json"
NOTES = _SCRATCH + "/playtest_notes.txt"


def _player(world):
    return world.entities["player"]


def _show(world):
    p = _player(world)
    print(engine.render(world, p))
    print("\nactions:")
    opts = engine.affordance_menu(world, p)
    for i, o in enumerate(opts):
        print(f"  {i}: {o.label}")
    return opts


def _save(world):
    with open(STATE, "wb") as f:
        pickle.dump(world, f)


def _load():
    with open(STATE, "rb") as f:
        return pickle.load(f)


def cmd_new(args):
    world, _ = scene()
    _save(world)
    _show(world)


def cmd_step(args):
    world = _load()
    opts = engine.affordance_menu(world, _player(world))
    if not (0 <= args.choice < len(opts)):
        print(f"invalid choice {args.choice} (0..{len(opts)-1})")
        return
    res = engine.step(world, opts[args.choice])
    _save(world)
    print(f">>> {opts[args.choice].label}")
    for line in res["narrative"]:
        print("   ", line)
    print()
    if engine.is_over(world):
        print(engine.render(world, _player(world)))
        print("\n*** You have died. Game over. Run `new` to restart. ***")
    else:
        _show(world)
    if res["violations"]:
        print("\n!! INCOHERENCE:", res["violations"])


def cmd_do(args):
    """Free-text action on the saved game: python -m src.core do "pour oil on the door" """
    from . import composer
    world = _load()
    res = engine.free_text(world, _player(world), " ".join(args.text), composer.MockPicker())
    _save(world)
    for line in res["narrative"]:
        print("   ", line)
    print()
    if engine.is_over(world):
        print("*** You have died. Game over. Run `new` to restart. ***")
    else:
        _show(world)
    if res["violations"]:
        print("!! INCOHERENCE:", res["violations"])


def cmd_play(args):
    world, _ = scene()
    while True:
        opts = _show(world)
        raw = input("\n> ").strip()
        if raw in ("q", "quit", "exit"):
            break
        if not raw.isdigit() or not (0 <= int(raw) < len(opts)):
            print("pick a number.")
            continue
        res = engine.step(world, opts[int(raw)])
        print()
        for line in res["narrative"]:
            print("   ", line)
        if res["violations"]:
            print("!! INCOHERENCE:", res["violations"])
        if engine.is_over(world):
            print("\n*** You have died. Game over. ***")
            break


def cmd_auto(args):
    rng = random.Random(args.seed)
    world, _ = scene(seed=args.seed)
    stats = {"turns": 0, "deaths": 0, "fires": 0, "collapses": 0, "violations": 0}
    epoch = 0
    seen = set()
    for _ in range(args.turns):
        opts = engine.affordance_menu(world, _player(world))
        res = engine.step(world, rng.choice(opts))
        stats["turns"] += 1
        for v in res["violations"]:
            stats["violations"] += 1
            print(f"!! INCOHERENCE at t{world.tick}: {v}")
            print("   recent log:", [e.story() for e in world.log[-6:]])
        if "guard-dead" not in seen and "dead" in world.entities["guard"].tags:
            seen.add("guard-dead"); stats["deaths"] += 1
        if "beam-gone" not in seen and any("collapsed" in c.tags for c in world.cells.values()):
            seen.add("beam-gone"); stats["collapses"] += 1
        if any("on_fire" in c.tags for c in world.cells.values()):
            stats["fires"] += 1
        # reseed once the player dies or the scene is spent, to keep finding chains
        spent = "dead" in world.entities["guard"].tags and any("collapsed" in c.tags for c in world.cells.values())
        if engine.is_over(world) or spent:
            epoch += 1
            world, _ = scene(seed=args.seed + epoch)
            seen.clear()
    print("\n=== autoplay report ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("  verdict:", "COHERENT (no invariant violations)" if stats["violations"] == 0
          else f"{stats['violations']} VIOLATIONS — investigate above")


# ── playtest session: an agent (Haiku subagent) plays turn-by-turn while the
# selector harvests sim-blessed protoreasoning traces into a persisted corpus.
# Sessions are TAG-NAMESPACED so many agents can play in PARALLEL without clobbering
# each other's world/notes; each writes its own corpus SHARD (no write races), and
# `merge` pools every shard into the master CORPUS. Diverse intents → diverse
# traces → a corpus that can actually correct the model's degenerate priors. ─────
def _paths(tag):
    t = "".join(c for c in tag if c.isalnum() or c in "-_") or "default"
    return (f"{_SCRATCH}/pt_{t}_world.pkl", f"{_SCRATCH}/pt_{t}_notes.txt",
            f"{_SCRATCH}/pt_{t}_shard.json")


def _dm_prompt(world, dc):
    """Surface a social choice-point for the DM (a big model). It SELECTS a typed
    stance — the line is the sim's; the DM never free-writes dialogue."""
    from .social import _voiced
    c, npc = dc.ctx, world.entities[dc.ctx["npc_id"]]
    feel = ", ".join(sorted(t for t in c["tags"] if t in
                            {"afraid", "hostile", "compliant", "calm"})) or "calm"
    rapport = ("loathes you" if c["disposition"] < -0.2 else
               "has warmed to you" if c["disposition"] > 0.2 else "barely knows you")
    said = f'You said: "{c["utterance"]}"' if c.get("utterance") else f"You chose to {c['intent']}"
    opts = "\n".join(f"  {i} {o.id}: \"{_voiced(o, npc)}\"" for i, o in enumerate(dc.options))
    return (f"‹DM CHOICE NEEDED›\n{said}\n{npc.name} — {feel}; composure "
            f"{c['composure']:.0f}; {rapport}.\noptions:\n{opts}\n"
            f"# As DM, pick the in-character reaction: session choose --tag <tag> <n>")


def _sess_setup(brain, shard):
    """Wire the selector + attach this session's own corpus SHARD as the live
    datastore (so parallel sessions never race on one file)."""
    import os

    from . import composer, knn, social
    social.use_brain(brain)
    social.DATASTORE = knn.Datastore.load(shard) if os.path.exists(shard) else knn.Datastore()
    return (composer.TorchPicker() if brain == "torch" else composer.MockPicker()), social


def _sess_scene(world, social):
    p, out = _player(world), [engine.render(world, _player(world)), "", "actions:"]
    out += [f"  {i}: {o.label}" for i, o in enumerate(engine.affordance_menu(world, p))]
    out.append(f"\n[corpus: {sum(len(v) for v in social.DATASTORE.buckets.values())} traces harvested]")
    return "\n".join(out)


def cmd_session(args):
    import glob
    import json
    import os
    from . import knn
    tag = getattr(args, "tag", "default")
    SESS_, NOTES_, SHARD_ = _paths(tag)

    if args.action == "merge":                 # pool every session shard into the master corpus
        master = knn.Datastore.load(CORPUS) if os.path.exists(CORPUS) else knn.Datastore()
        shards = sorted(glob.glob(f"{_SCRATCH}/pt_*_shard.json"))
        added = 0
        for sh in shards:
            for key, anchors in json.load(open(sh)).items():
                master.buckets.setdefault(key, []).extend(anchors)
                added += len(anchors)
        master.save(CORPUS)
        total = sum(len(v) for v in master.buckets.values())
        return print(f"# merged {len(shards)} shard(s): +{added} traces → {total} total in {CORPUS}\n"
                     f"# buckets: {json.dumps({k: len(v) for k, v in master.buckets.items()}, indent=0)}")

    if args.action == "note":                  # the agent logs a fun/coherence gap
        open(NOTES_, "a").write(" ".join(args.text) + "\n")
        return print("noted.")

    if args.action == "start":
        from . import seed as _sm
        picker, social = _sess_setup(args.brain, SHARD_)
        mk = _sm.generate if getattr(args, "gen", False) else getattr(_sm, getattr(args, "scene", "guildhall"))
        world, _ = mk(seed=args.seed)
        if getattr(args, "pack", ""):          # layer LitRPG system pack(s) onto the world
            from . import reactions
            for pk in args.pack.split(","):
                world.packs.append(reactions.load_pack(pk))
        for f in (NOTES_, SHARD_, SESS_.replace("_world.pkl", "_pending.json"),
                  SESS_.replace("_world.pkl", "_dm.jsonl")):   # fresh session: clear stale state
            if os.path.exists(f):
                os.remove(f)
        social.DATASTORE = knn.Datastore()     # start this session's shard empty
    else:
        with open(SESS_, "rb") as f:
            d = pickle.load(f)
        world, args.brain = d["world"], d["brain"]
        picker, social = _sess_setup(args.brain, SHARD_)

    if args.action == "end":
        n = sum(len(v) for v in social.DATASTORE.buckets.values())
        notes = open(NOTES_).read().strip() if os.path.exists(NOTES_) else "(none)"
        return print(f"# session '{tag}' ended. {n} traces in shard → {SHARD_}\n"
                     f"# (run `session merge` to pool all shards into the master corpus)\n"
                     f"\n=== playtester notes ===\n{notes}")

    p = _player(world)
    PEND_ = SESS_.replace("_world.pkl", "_pending.json")   # a deferred DM choice-point
    DM_ = SESS_.replace("_world.pkl", "_dm.jsonl")         # harvested big-model DM picks
    if args.action == "start":
        print(f"# playtest session '{tag}' started (brain={args.brain}, seed={args.seed})")
    else:
        # a `choose` resumes a deferred turn with one more DM answer appended;
        # `do`/`step` start fresh. The agent brain replays deterministically.
        pend = json.load(open(PEND_)) if args.action == "choose" else \
            {"cmd": args.action, "text": " ".join(getattr(args, "text", []) or []),
             "n": getattr(args, "n", 0), "answers": []}
        if args.action == "choose":
            pend["answers"] = pend["answers"] + [args.n]
        if args.brain == "agent":
            social.use_brain("agent", answers=pend["answers"])
        try:
            if pend["cmd"] == "step":
                opts = engine.affordance_menu(world, p)
                if not (0 <= pend["n"] < len(opts)):
                    return print(f"invalid option {pend['n']} (0..{len(opts)-1})")
                print(f">>> {opts[pend['n']].label}")
                res = engine.step(world, opts[pend["n"]])
            else:                              # do / choose → replay the text
                res = engine.free_text(world, p, pend["text"], picker)
        except social.DMChoice as dc:
            json.dump(pend, open(PEND_, "w"))
            return print(_dm_prompt(world, dc))
        if os.path.exists(PEND_):
            os.remove(PEND_)
        if args.brain == "agent" and social.BRAIN.log:     # bank the DM's picks
            with open(DM_, "a") as f:
                for row in social.BRAIN.log:
                    f.write(json.dumps(row) + "\n")

    with open(SESS_, "wb") as f:               # persist world + brain
        pickle.dump({"world": world, "brain": args.brain}, f)
    social.DATASTORE.save(SHARD_)              # this session's harvested traces
    if args.action != "start":
        for line in res["narrative"]:
            print("   ", line)
        if res["violations"]:
            print("!! INCOHERENCE:", res["violations"])
        print()
    print("*** You have died. Game over. ***" if engine.is_over(world) else _sess_scene(world, social))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="src.core")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("new").set_defaults(fn=cmd_new)
    sp = sub.add_parser("step"); sp.add_argument("choice", type=int); sp.set_defaults(fn=cmd_step)
    sub.add_parser("play").set_defaults(fn=cmd_play)
    sa = sub.add_parser("auto")
    sa.add_argument("--turns", type=int, default=2000)
    sa.add_argument("--seed", type=int, default=0)
    sa.set_defaults(fn=cmd_auto)
    sd = sub.add_parser("do")
    sd.add_argument("text", nargs="+")
    sd.set_defaults(fn=cmd_do)

    se = sub.add_parser("eval")
    se.add_argument("--brain", choices=["mock", "torch"], default="mock")
    se.add_argument("--band", default="")
    se.add_argument("--debug", action="store_true")
    se.set_defaults(fn=lambda a: __import__("src.core.eval_composer", fromlist=["run"]).run(
        a.brain, a.band, debug=a.debug))

    ss = sub.add_parser("session", help="agent playtest loop; harvests traces to a persisted corpus")
    ssub = ss.add_subparsers(dest="action", required=True)
    tagp = argparse.ArgumentParser(add_help=False)     # --tag namespaces parallel sessions
    tagp.add_argument("--tag", default="default")
    st0 = ssub.add_parser("start", parents=[tagp]); st0.add_argument("--brain", choices=["mock", "torch", "agent"], default="torch")
    st0.add_argument("--seed", type=int, default=0)
    st0.add_argument("--gen", action="store_true", help="use the PROCEDURAL room for this seed (generalization runs)")
    st0.add_argument("--pack", default="", help="comma-separated LitRPG packs to layer on (e.g. cultivation)")
    st0.add_argument("--scene", default="guildhall", choices=["guildhall", "vault", "dungeon", "farm"])
    st1 = ssub.add_parser("step", parents=[tagp]); st1.add_argument("n", type=int)
    st2 = ssub.add_parser("do", parents=[tagp]); st2.add_argument("text", nargs="+")
    st3 = ssub.add_parser("note", parents=[tagp]); st3.add_argument("text", nargs="+")
    st4 = ssub.add_parser("choose", parents=[tagp]); st4.add_argument("n", type=int)  # DM stance pick
    ssub.add_parser("end", parents=[tagp])
    ssub.add_parser("merge")                           # pool all shards → master corpus
    ss.set_defaults(fn=cmd_session)

    sw = sub.add_parser("web")
    sw.add_argument("--port", type=int, default=8000)
    sw.add_argument("--brain", choices=["mock", "torch"], default="mock")
    sw.add_argument("--voice", choices=["none", "reyna"], default="none")
    sw.set_defaults(fn=lambda a: __import__("src.core.webserver", fromlist=["serve"]).serve(a.port, a.brain, a.voice))
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
