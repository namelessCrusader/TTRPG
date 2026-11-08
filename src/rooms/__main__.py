"""Run the inn through the mind (docs/sim-mind-design.md 3.5, step M5).

    python -m src.rooms run --seeds 6 --turns 300 --chooser rules
    python -m src.rooms run --seeds 6 --menu two         # scope, then act (src/mind/scope.py)
    python -m src.rooms run --compat --seeds 3            # the lab's nights, byte for byte
    python -m src.rooms run --play Fenn --seed 0          # play Fenn: type a number at each choice
    python -m src.rooms session start --as Fenn --seed 0  # play over many calls: an agent playtest
    python -m src.rooms session pick 3 [5 ...]
    python -m src.rooms session show | end

RUN. Each night (seed) goes through the loop of the mind (src/mind/loop.py)
in reactive mode: every NPC has the rules (or --chooser random), and the
model is never asked. Two files for each night go to --out:
    seed<k>.events.jsonl      the events, in the lab's format and with the
                              lab's verbs. lab/judge/signals.py reads them
                              as it reads the lab's nights.
    seed<k>.decisions.jsonl   one row for each job started: the request, the
                              menu, the cut, the pick and its Why.
With --menu two, each NPC chooses in small questions: a scope, then an act
in it (src/mind/scope.py; a third question only when "something else" is
picked on a menu too big for two); its decision row holds every level, and
the event's options are the full menu, with one draw over it. With --compat, a night is the lab's night (backend.night): every option is
on the menu, and the lab's chooser draws with the world's own rng. Then the
events are the events of the inn goldens, byte for byte, and the decision
rows are the backend's rows (the Why of each choice).

PLAY. You are one person of the inn. At each of its choices you read the
scene text and the full menu, in groups (a player's menu is never cut), and
you type the number of an option. More numbers make a chain: each option
starts when the one before it is done. When the next option cannot start
then, or a job of the chain is not done, the chain stops, and the screen
says so. q stops. The other people have the rules.

SESSION. The same game over many calls, for an agent that plays one choice
at a time. The state is the seed and the picks so far, in a JSON file in
--dir. Each call runs the night again from its start with those picks, up to
the player's next choice. The choosers draw from seeded generators, so the
same picks give the same night. When the night no longer asks the same
questions (the code changed), the session says so and stops.

The default folder for the files is logs/rooms in the repository. Git does
not keep logs/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import textwrap
import time
from pathlib import Path

from src.mind.loop import MENUS, NPC, Mind, MindConfig, run as run_turns

from .backend import RoomsBackend, jsonl, night as compat_night
from .world import CAST

SETTING = "It is night at the village inn"
LOGS = Path(__file__).resolve().parents[2] / "logs" / "rooms"
CAST_NAMES = tuple(name for name, *_ in CAST)      # who a player can be: the bandit comes later
WIDTH = 100                                         # the width of the scene text on a screen


class Stop(Exception):
    """The player stops the game, or a session reaches the player's next
    choice (`req`, the request of that choice)."""

    def __init__(self, req=None):
        super().__init__()
        self.req = req


def one_night(seed, turns, chooser="rules", player=None, ask=None, menu="flat"):
    """One night of the inn through the mind, in reactive mode. Returns the
    backend and the mind."""
    b = RoomsBackend(seed, turns)
    mind = Mind(MindConfig(player=player, npc=chooser, seed=seed, setting=SETTING, menu=menu), ask=ask)
    run_turns(b, mind, turns)
    return b, mind


def _write(path, text):
    """Write a file whole: a run that stops leaves no half file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _player(name):
    if name not in CAST_NAMES:
        raise SystemExit(f"{name!r} is not a person of the inn. Play one of: {', '.join(CAST_NAMES)}")
    return name


# ---------------------------------------------------------------- what a player sees
def screen(req, turn):
    """One choice of the player: the scene text, the question, and the menu
    in its groups."""
    lines = [f"--- turn {turn} ---", *textwrap.wrap(req.situation, WIDTH), req.question]
    group = None
    for i, (text, g) in enumerate(zip(req.options, req.groups)):
        if g != group:
            lines.append(f"  {g}")
            group = g
        lines.append(f"    {i:>2}  {text}")
    return "\n".join(lines)


def news(b, mind, who, since, asked=False):
    """What became of the jobs of `who` after job event number `since`: one
    line for each job that ended. `asked`: the player must choose now. Then,
    when its last job had a chain, the chain stops here (its next option
    cannot start now, or the job was not done), and a line says so."""
    texts = {r["job"]: r["text"] for r in mind.rows}
    out = []
    for e in b.job_events[since:]:
        if e.who != who or e.phase == "start":
            continue
        if e.phase == "done":
            out.append(f"  ({e.text})")
        elif e.phase == "failed":
            out.append(f"  ({who} could not {texts[e.job]}: {e.reason})")
        else:
            out.append(f"  ({who} was stopped and did not {texts[e.job]})")
    job = mind.jobs.get(who)
    if asked and job is not None and job.then:
        out.append(f"  ({who} cannot {job.then[0].text} now: the chain stops)" if job.status == "done"
                   else "  (the chain stops)")
    return out


def _say(out, lines):
    if isinstance(lines, str):
        lines = [lines]
    for x in lines:
        out.write(x + "\n")
    out.flush()


# ---------------------------------------------------------------- run
def cmd_run(a):
    if a.play:
        if a.compat:
            raise SystemExit("--play and --compat do not go together: compat mode has no player")
        return play(a.seed, a.turns, _player(a.play), a.chooser, sys.stdin, sys.stdout)
    if a.compat and a.menu != "flat":
        raise SystemExit("--compat is the lab's night: its menu is flat")
    out = Path(a.out) if a.out else LOGS / (f"compat-{a.chooser}" if a.compat else
                                            a.chooser + ("" if a.menu == "flat" else f"-{a.menu}"))
    out.mkdir(parents=True, exist_ok=True)
    for seed in range(a.seed, a.seed + a.seeds):
        t0 = time.time()
        if a.compat:
            b = compat_night(seed, a.turns, a.chooser)
            rows, calls = b.decisions, 0
        else:
            b, mind = one_night(seed, a.turns, a.chooser, menu=a.menu)
            rows, calls = mind.rows, sum(mind.remote.values())
        _write(out / f"seed{seed}.events.jsonl", jsonl(b.world.log))
        _write(out / f"seed{seed}.decisions.jsonl", jsonl(rows))
        print(f"  seed {seed}: {len(b.world.log)} events, {len(rows)} decisions, {calls} model calls,"
              f" {time.time() - t0:.1f}s", flush=True)
    print(out)


def play(seed, turns, who, chooser, inp, out):
    """Play `who` for one night: read a choice from `inp` at each of its
    decisions, and write the screens to `out`. Returns the backend and the
    mind."""
    b = RoomsBackend(seed, turns)
    since, holder = [0], {}

    def ask(req):
        _say(out, news(b, holder["mind"], who, since[0], asked=True))
        since[0] = len(b.job_events)
        _say(out, screen(req, b.world.t))
        n = len(req.options)
        while True:
            out.write("your choice: ")
            out.flush()
            line = inp.readline()
            if not line or line.strip().lower() in ("q", "quit"):
                raise Stop()
            try:
                picks = [int(x) for x in line.split()]
            except ValueError:
                picks = []
            if picks and all(0 <= i < n for i in picks):
                return picks
            _say(out, f"type the number of an option, from 0 to {n - 1}. More numbers make a chain. q stops.")

    mind = holder["mind"] = Mind(MindConfig(player=who, npc=chooser, seed=seed, setting=SETTING), ask=ask)
    try:
        run_turns(b, mind, turns)
    except Stop:
        _say(out, "\nstopped.")
        return b, mind
    _say(out, news(b, mind, who, since[0]))
    _say(out, f"\nthe night is over (turn {b.world.t}).")
    return b, mind


# ---------------------------------------------------------------- session
def _state_file(a):
    tag = "".join(c for c in a.tag if c.isalnum() or c in "-_") or "default"
    return Path(a.dir) / f"session-{tag}.json"


def _digest(req):
    """The question a pick answered: the same request gives the same digest."""
    return hashlib.sha256(json.dumps([req.id, req.situation, list(req.options)]).encode()).hexdigest()[:16]


def replay(state):
    """Run the night of a session again with its picks, up to the player's
    next choice. Returns (backend, mind, the request of that choice, or None
    when the night ended first, the number of job events at each choice)."""
    b = RoomsBackend(state["seed"], state["turns"])
    picks, marks = list(state["picks"]), []

    def ask(req):
        k = len(marks)
        marks.append(len(b.job_events))
        if k == len(state["picks"]):
            raise Stop(req)
        if _digest(req) != state["asked"][k]:
            raise SystemExit(f"this session no longer replays: choice {k} is not the question it was"
                             " (the code changed). Start a new session.")
        return picks[k]

    mind = Mind(MindConfig(player=state["as"], npc=state["npc"], seed=state["seed"], setting=SETTING),
                ask=ask)
    try:
        run_turns(b, mind, state["turns"])
    except Stop as s:
        return b, mind, s.req, marks
    return b, mind, None, marks


def _show(a, state, got):
    b, mind, req, marks = got
    # what happened after the last pick: from the choice before the one that waits, or
    # from the last choice when the night ended first
    done = marks if req is None else marks[:-1]
    lines = news(b, mind, state["as"], done[-1] if done else 0, asked=req is not None)
    if req is None:
        lines.append(f"# the night is over (turn {b.world.t}). End the session: "
                     f"python -m src.rooms session end --tag {a.tag}")
    else:
        lines += [screen(req, b.world.t),
                  f"# pick: python -m src.rooms session pick <n> [<n> ...] --tag {a.tag}"]
    print("\n".join(lines))


def cmd_session(a):
    f = _state_file(a)
    if a.action == "start":
        state = {"seed": a.seed, "turns": a.turns, "as": _player(a.who), "npc": a.chooser,
                 "picks": [], "asked": []}
        f.parent.mkdir(parents=True, exist_ok=True)
        _write(f, json.dumps(state))
        print(f"# session {a.tag!r}: you are {state['as']}, seed {a.seed}, {a.turns} turns")
        return _show(a, state, replay(state))
    if not f.exists():
        raise SystemExit(f"no session {a.tag!r} in {f.parent}: start one with `session start --as NAME`")
    state = json.loads(f.read_text())
    got = replay(state)
    if a.action == "show":
        return _show(a, state, got)
    if a.action == "end":
        b, mind = got[0], got[1]
        stem = f.name[:-len(".json")]
        _write(f.with_name(stem + ".events.jsonl"), jsonl(b.world.log))
        _write(f.with_name(stem + ".decisions.jsonl"), jsonl(mind.rows))
        f.unlink()
        n = len(state["picks"])
        print(f"# session {a.tag!r} ended after {n} choice{'' if n == 1 else 's'}, at turn {b.world.t}."
              f" The night is in {f.with_name(stem + '.events.jsonl')}")
        return
    req = got[2]                            # pick
    if req is None:
        raise SystemExit("the night is over: there is no choice to make")
    if not all(0 <= i < len(req.options) for i in a.n):
        raise SystemExit(f"type the number of an option, from 0 to {len(req.options) - 1}")
    state["picks"].append(list(a.n))
    state["asked"].append(_digest(req))
    got = replay(state)
    _write(f, json.dumps(state))
    print(">>> " + "; then ".join(req.options[i] for i in a.n))
    _show(a, state, got)


# ---------------------------------------------------------------- the command line
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.rooms", description="The inn, through the mind.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run nights of the inn, and write their events and decisions")
    r.add_argument("--seeds", type=int, default=6, help="how many nights")
    r.add_argument("--seed", type=int, default=0, help="the seed of the first night")
    r.add_argument("--turns", type=int, default=300)
    r.add_argument("--chooser", choices=sorted(NPC), default="rules", help="the chooser of every NPC")
    r.add_argument("--compat", action="store_true", help="the lab's nights, byte for byte")
    r.add_argument("--menu", choices=MENUS, default="flat",
                   help="flat: one question; two: a scope, then an act in it")
    r.add_argument("--out", help="the folder for the files (default: logs/rooms/<chooser>)")
    r.add_argument("--play", metavar="NAME", help="play this person for one night (seed --seed)")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("session", help="play one choice for each call: an agent playtest")
    ssub = s.add_subparsers(dest="action", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tag", default="default", help="the name of the session")
    common.add_argument("--dir", default=str(LOGS), help="the folder of the session files")
    st = ssub.add_parser("start", parents=[common])
    st.add_argument("--as", dest="who", required=True, help=f"who you play: {', '.join(CAST_NAMES)}")
    st.add_argument("--seed", type=int, default=0)
    st.add_argument("--turns", type=int, default=300)
    st.add_argument("--chooser", choices=sorted(NPC), default="rules", help="the chooser of every NPC")
    ssub.add_parser("pick", parents=[common]).add_argument("n", type=int, nargs="+")
    ssub.add_parser("show", parents=[common])
    ssub.add_parser("end", parents=[common])
    s.set_defaults(fn=cmd_session)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    main()
