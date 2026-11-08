"""The verb catalog: what a mind can choose to do, as data (`data/verbs.yaml`).

A VERB is a task ("take", "walk to"), not a limb move. For each verb the
catalog gives the roles it fills and the tests on them, the capabilities it
needs, the drives it serves, its words (the option text and the "told"
reports), how long it takes, what interrupts it, its flags, and one program
for each backend that can run it. A backend never offers a verb that has no
program for it.

THE CATALOG KNOWS NO BACKEND. What a backend can run (the verbs of the inn,
the limbs and step tests of voxel) is written in the file, under `backends`.
A program is checked against that declaration, by its shape: a mapping names
one native verb, and a list gives the steps of a limb program.

`load()` reads the file and checks it. It collects every problem, and one
`CatalogError` lists all of them, so that an author can correct the file in
one pass. Only a problem with the SHAPE of the file (a top-level part that is
missing, or that is not a mapping or a list) stops the checks early, because
then the verbs cannot be read against the top-level names. The checks of the
whole catalog (one null act, every native verb run) need every verb, so they
run only when each verb passes.

These are the checks of the design (docs/sim-mind-design.md 3.5, first in
`check_catalog_v2.py`), and some more that keep the menu rules true:
- every name a template uses is a declared role, and `text` uses every role
  that is not optional, so two options of one verb never have the same text;
- a verb that can run has a `told` report and a duration on each backend
  that runs it, and a loud or violent verb also has a `told_start`;
- every capability, drive, flag and interrupt class that a verb names is
  declared; a capability has its words (`says`), and a test on each backend
  where a verb needs it;
- exactly one verb is the null act, it has no roles, and every backend runs it;
- every native verb that a backend declares is run by some catalog verb.

The catalog is read-only. A backend program, or a test of a capability or a
drive, is handed out as a new copy, so that a backend can change its copy (a
job keeps its own steps) without a change to the catalog.
"""
from __future__ import annotations

import copy
import os
import re
import string
from dataclasses import dataclass
from types import MappingProxyType

from .api import Option, Ref

PATH = os.path.join(os.path.dirname(__file__), "data", "verbs.yaml")

TOP = ("version", "groups", "backends", "capabilities", "drives", "flags_known",
       "interrupts_known", "verbs")
VERB_KEYS = ("id", "group", "null_act", "roles", "needs", "serves", "hint", "text", "text_with",
             "told", "told_with", "told_start", "duration", "commit_after", "interrupt_on",
             "flags", "backends")
ROLE_KEYS = ("kind", "optional", "is", "not", "held_by", "cap", "cap_any", "reach")
KINDS = ("person", "thing", "place")
BACKEND_KEYS = ("verbs", "limbs", "step_tests")
STEP_KEYS = ("limb", "match", "until", "aim", "flee")
# how a limb step finds its row on a menu: by payload, never by label
STEP_MATCH = ("goal", "cell", "eid", "who")
TEMPLATES = ("text", "text_with", "told", "told_with", "told_start")
NAME = re.compile(r"[a-z][a-z_]*\Z")


class CatalogError(ValueError):
    """The catalog breaks one or more rules. `problems` lists every one."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("the verb catalog has %d problem(s):\n  " % len(self.problems)
                         + "\n  ".join(self.problems))


@dataclass(frozen=True)
class Role:
    """A role of a verb, and the tests on what fills it."""
    name: str
    kind: str                       # person | thing | place
    optional: bool = False
    is_: tuple[str, ...] = ()       # states it must have ("down")
    not_: tuple[str, ...] = ()      # states it must not have ("fixed", "held")
    held_by: str | None = None      # "self", or the role of the person who holds it
    cap: tuple[str, ...] = ()       # capabilities it must have, all of them
    cap_any: tuple[str, ...] = ()   # capabilities it must have, one of them
    reach: str | None = None        # a distance band that the backend measures


@dataclass(frozen=True)
class Capability:
    """A test on a thing, or on the actor itself, and the words for it."""
    name: str
    says: str                       # what a chooser reads: "something with a sharp edge"
    of_self: bool                   # a test on the actor ("a free hand"), not on a role
    tested_on: tuple[str, ...]      # the backends that have a test for it


@dataclass(frozen=True)
class Verb:
    """One verb of the catalog. `runs_on` lists the backends that have a
    program for it, and `durations` gives its duration on each of them."""
    id: str
    group: str
    roles: tuple[Role, ...]
    text: str
    null_act: bool = False
    needs: tuple[str, ...] = ()
    serves: tuple[str, ...] = ()
    hint: str | None = None
    text_with: str | None = None
    told: str | None = None
    told_with: str | None = None
    told_start: str | None = None
    durations: tuple[tuple[str, int | str], ...] = ()   # (backend, turns | ticks | "route")
    commit_after: float | None = None
    interrupt_on: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    runs_on: tuple[str, ...] = ()

    def role(self, name):
        return next((r for r in self.roles if r.name == name), None)

    def duration(self, backend):
        return dict(self.durations).get(backend)

    def option_text(self, names):
        """The words of an option, from the side of the actor: "strike Tom
        with the axe". `names` maps each filled role to its name. When an
        optional role is filled, the words come from `text_with`."""
        return self._fill(self.text_with if self._optional_in(names) else self.text, names)

    def told_text(self, actor, names, start=False):
        """The report of the act as others tell it: "Fenn struck Tom". With
        `start`, the report of the act as it starts."""
        if start:
            tpl = self.told_start
        else:
            tpl = self.told_with if self.told_with and self._optional_in(names) else self.told
        if tpl is None:
            raise ValueError(f"verb {self.id} has no {'told_start' if start else 'told'} report")
        return self._fill(tpl, {**names, "actor": actor})

    def _optional_in(self, names):
        return any(r.optional and names.get(r.name) for r in self.roles)

    def _fill(self, tpl, names):
        try:
            return tpl.format_map(names)
        except KeyError as e:
            raise ValueError(f"verb {self.id}: no name for role {e.args[0]!r} in {tpl!r}") from None


def load(path=PATH):
    """Read the catalog file and check it."""
    import yaml         # the one import of src/mind outside the standard library: only to read the file
    with open(path, encoding="utf-8") as f:
        return Catalog(yaml.safe_load(f))


def _fields(tpl):
    """The names a template uses, or None when it is not a plain template
    (a brace that does not close, a format spec, an attribute, an index)."""
    try:
        parts = list(string.Formatter().parse(tpl))
    except ValueError:
        return None
    out = set()
    for _, name, spec, conv in parts:
        if name is None:
            continue
        if spec or conv or not NAME.match(name):
            return None
        out.add(name)
    return out


class Catalog:
    """The checked catalog. Build it with `load()`, or from a parsed mapping
    with `Catalog(data)`."""

    def __init__(self, data):
        self._bad, self._broken = [], False
        if not isinstance(data, dict):
            raise CatalogError(["the file is not a mapping"])
        self._raw = copy.deepcopy(data)
        self._top(data)
        if self._broken:                # the verbs are checked against the top-level names
            raise CatalogError(self._bad)
        verbs, seen = {}, set()
        for v in data["verbs"]:
            verb = self._verb(v, seen)
            if verb is not None:
                verbs[verb.id] = verb
        if len(verbs) == len(data["verbs"]):    # the checks of the whole need every verb
            self._programs = {v["id"]: v["backends"] for v in self._raw["verbs"]}
            self._whole(verbs)
        if self._bad:
            raise CatalogError(self._bad)
        self.verbs = MappingProxyType(verbs)
        self.null_act = next(v for v in verbs.values() if v.null_act)
        # canonical order: the group, in the order of `groups`; then the order of the file
        ranked = sorted(verbs, key=lambda k: self.groups.index(verbs[k].group))
        self._rank = {k: i for i, k in enumerate(ranked)}

    # ------------------------------------------------------------ checks
    def _say(self, msg, shape=False):
        """Record a problem. `shape`: the problem breaks the shape of the
        file, so the verbs cannot be checked."""
        self._bad.append(msg)
        self._broken |= shape

    def _list(self, x, what, unique=True, shape=False):
        """A list of names. A problem when it is not one. None is no names,
        except for a part of the file's shape, which must be there."""
        if x is None and not shape:
            return ()
        if not isinstance(x, list) or not all(isinstance(s, str) for s in x):
            self._say(f"{what} is not a list of names: {x!r}", shape)
            return ()
        if unique and len(set(x)) != len(x):
            self._say(f"{what} names something more than once: {x}")
        return tuple(x)

    def _top(self, d):
        for k in d:
            if k not in TOP:
                self._say(f"unknown top-level key {k!r}")
        for k in TOP:
            if k not in d:
                self._say(f"no top-level key {k!r}", shape=True)
        if self._broken:
            return
        if not isinstance(d["version"], int) or isinstance(d["version"], bool) or d["version"] < 1:
            self._say(f"version is not a positive integer: {d['version']!r}")
        self.version = d["version"]
        self.groups = self._list(d["groups"], "groups", shape=True)
        self.flags_known = self._list(d["flags_known"], "flags_known", shape=True)
        self.interrupts_known = self._list(d["interrupts_known"], "interrupts_known", shape=True)
        self.backends = ()
        if not isinstance(d["backends"], dict) or not d["backends"]:
            self._say("backends is not a mapping of backend names", shape=True)
        else:
            self.backends = tuple(d["backends"])
            for be, spec in d["backends"].items():
                if not isinstance(spec, dict) or set(spec) - set(BACKEND_KEYS):
                    self._say(f"backend {be!r}: the declaration has keys other than {BACKEND_KEYS}",
                              shape=True)
                    continue
                for k in spec:
                    self._list(spec[k], f"backend {be!r} {k}", shape=True)
                if ("limbs" in spec) != ("step_tests" in spec):
                    self._say(f"backend {be!r}: limbs and step_tests go together", shape=True)
        caps = {}
        if not isinstance(d["capabilities"], dict):
            self._say("capabilities is not a mapping", shape=True)
        else:
            for name, spec in d["capabilities"].items():
                cap = self._capability(name, spec)
                if cap is not None:
                    caps[name] = cap
        self.capabilities = MappingProxyType(caps)
        self.drives = ()
        if not isinstance(d["drives"], dict):
            self._say("drives is not a mapping", shape=True)
        else:
            self.drives = tuple(d["drives"])
            for name, spec in d["drives"].items():
                if not isinstance(spec, dict) or set(spec) - set(self.backends):
                    self._say(f"drive {name!r}: its keys are not backend names")
                elif any(t is not None and not isinstance(t, dict) for t in spec.values()):
                    self._say(f"drive {name!r}: a test is not a mapping or null")
        if not isinstance(d["verbs"], list) or not all(isinstance(v, dict) for v in d["verbs"]):
            self._say("verbs is not a list of mappings", shape=True)

    def _capability(self, name, spec):
        if not isinstance(spec, dict):
            # a verb that names it would read as naming an unknown capability
            self._say(f"capability {name!r} is not a mapping", shape=True)
            return None
        for k in spec:
            if k not in ("says", "self") + self.backends:
                self._say(f"capability {name!r}: unknown key {k!r}")
        says = spec.get("says")
        if not isinstance(says, str) or not says.strip():
            self._say(f"capability {name!r} has no words (says)")
        if not isinstance(spec.get("self", False), bool):
            self._say(f"capability {name!r}: self is not true or false")
        tested = []
        for be in self.backends:
            t = spec.get(be)
            if t is None:
                continue
            if not isinstance(t, dict):
                self._say(f"capability {name!r}: the {be} test is not a mapping or null")
                continue
            tested.append(be)
            if "edge_max_m2" in t and t.get("edge_gt_m2") is None:
                # fill(edge=...) writes 0.0 for blunt, and 0.0 is below any maximum
                self._say(f"capability {name!r}: an edge test with no edge_gt_m2 admits edge=0 (blunt)")
        return Capability(name, says if isinstance(says, str) else "", bool(spec.get("self")),
                          tuple(tested))

    def _verb(self, v, seen):
        vid = v.get("id")
        where = f"verb {vid!r}"
        before = len(self._bad)
        if None in v:
            self._say(f"{where}: a key named null reads as None in YAML; write null_act")
        for k in v:
            if k is not None and k not in VERB_KEYS:
                self._say(f"{where}: unknown field {k!r}")
        if not isinstance(vid, str) or not NAME.match(vid):
            self._say(f"{where}: the id is not a lower-case name")
            return None
        if vid in seen:
            self._say(f"{where}: the id occurs more than once")
            return None
        seen.add(vid)
        if v.get("group") not in self.groups:
            self._say(f"{where}: group {v.get('group')!r} is not in groups")
        if not isinstance(v.get("null_act", False), bool):
            self._say(f"{where}: null_act is not true or false")
        roles = self._roles(where, v.get("roles"))
        names = {r.name for r in roles}
        caps = self.capabilities
        needs = self._list(v.get("needs"), f"{where} needs")
        for c in needs:
            if c not in caps:
                self._say(f"{where}: unknown capability {c!r} in needs")
            elif not caps[c].of_self:
                self._say(f"{where}: needs {c!r}, which is not a capability of the actor (self)")
        hint = v.get("hint")
        if hint is not None and (not isinstance(hint, str) or hint not in caps):
            self._say(f"{where}: unknown capability {hint!r} in hint")
        serves = self._list(v.get("serves"), f"{where} serves")
        for d in serves:
            if d not in self.drives:
                self._say(f"{where}: unknown drive {d!r}")
        flags = self._list(v.get("flags"), f"{where} flags")
        for f in flags:
            if f not in self.flags_known:
                self._say(f"{where}: unknown flag {f!r}")
        stops = self._list(v.get("interrupt_on"), f"{where} interrupt_on")
        for s in stops:
            if s not in self.interrupts_known:
                self._say(f"{where}: unknown interrupt class {s!r}")
        ca = v.get("commit_after")
        if ca is not None and (isinstance(ca, bool) or not isinstance(ca, (int, float)) or not 0 < ca <= 1):
            self._say(f"{where}: commit_after is not a fraction in (0, 1]")
        runs = self._programs_of(where, v.get("backends"), names)
        durations = self._durations(where, v.get("duration"), runs)
        for be in runs:
            for c in needs:
                if c in caps and be not in caps[c].tested_on:
                    self._say(f"{where}: runs on {be} and needs {c!r}, which has no {be} test")
            for r in roles:
                for c in r.cap:
                    if c in caps and be not in caps[c].tested_on:
                        self._say(f"{where}: runs on {be}, and role {r.name!r} needs {c!r},"
                                  f" which has no {be} test")
                if r.cap_any and not any(c in caps and be in caps[c].tested_on for c in r.cap_any):
                    self._say(f"{where}: runs on {be}, and no capability of role {r.name!r}"
                              f" cap_any has a {be} test")
        tpl = self._templates(where, v, roles)
        if runs and tpl.get("told") is None:
            self._say(f"{where}: it can run, but it has no told report")
        if runs and {"loud", "violent"} & set(flags) and tpl.get("told_start") is None:
            self._say(f"{where}: a loud or violent verb has a told_start report")
        if v.get("null_act"):
            if roles:
                self._say(f"{where}: the null act has no roles")
            if set(runs) != set(self.backends):
                self._say(f"{where}: the null act runs on every backend")
        if len(self._bad) > before:
            return None
        return Verb(vid, v["group"], roles, tpl["text"], bool(v.get("null_act")), needs, serves,
                    hint, tpl.get("text_with"), tpl.get("told"), tpl.get("told_with"),
                    tpl.get("told_start"), durations, ca, stops, flags, runs)

    def _roles(self, where, spec):
        if spec is None or not isinstance(spec, dict):
            self._say(f"{where}: roles is not a mapping (write roles: {{}} for none)")
            return ()
        out = []
        for name, r in spec.items():
            at = f"{where} role {name!r}"
            if not isinstance(name, str) or not NAME.match(name) or name == "actor":
                self._say(f"{at}: the name is not a lower-case name other than actor")
                continue
            if not isinstance(r, dict):
                self._say(f"{at}: not a mapping")
                continue
            for k in r:
                if k not in ROLE_KEYS:
                    self._say(f"{at}: unknown key {k!r}")
            if r.get("kind") not in KINDS:
                self._say(f"{at}: kind {r.get('kind')!r} is not one of {KINDS}")
            if not isinstance(r.get("optional", False), bool):
                self._say(f"{at}: optional is not true or false")
            held = r.get("held_by")
            if held is not None and held != "self" and (not isinstance(held, str) or held not in spec
                                                         or held == name):
                self._say(f"{at}: held_by {held!r} is not self or another role")
            cap = r.get("cap", [])
            cap = self._list([cap] if isinstance(cap, str) else cap, f"{at} cap")
            cap_any = self._list(r.get("cap_any"), f"{at} cap_any")
            for c in cap + cap_any:
                if c not in self.capabilities:
                    self._say(f"{at}: unknown capability {c!r}")
                elif self.capabilities[c].of_self:
                    self._say(f"{at}: {c!r} is a capability of the actor (self), not of a role")
            reach = r.get("reach")
            if reach is not None and not isinstance(reach, str):
                self._say(f"{at}: reach is not a name")
            out.append(Role(name, r.get("kind"), bool(r.get("optional")),
                            self._list(r.get("is"), f"{at} is"), self._list(r.get("not"), f"{at} not"),
                            held, cap, cap_any, reach))
        return tuple(out)

    def _programs_of(self, where, spec, roles):
        """The backends that run the verb. Each program is checked against the
        declaration of its backend."""
        if not isinstance(spec, dict):
            self._say(f"{where}: backends is not a mapping")
            return ()
        runs = []
        decl = self._raw["backends"]
        for be, prog in spec.items():
            if be not in self.backends:
                self._say(f"{where}: unknown backend {be!r}")
                continue
            if prog is None:
                continue
            runs.append(be)
            at = f"{where} on {be}"
            if isinstance(prog, dict):
                if "verbs" not in decl[be]:
                    self._say(f"{at}: {be} declares no verbs, so a program cannot name one")
                elif set(prog) != {"verb"} or prog["verb"] not in decl[be]["verbs"]:
                    self._say(f"{at}: the program is not {{verb: <one of the {be} verbs>}}: {prog!r}")
            elif isinstance(prog, list) and prog:
                if "limbs" not in decl[be]:
                    self._say(f"{at}: {be} declares no limbs, so a program cannot list steps")
                    continue
                for st in prog:
                    self._step(at, st, decl[be], roles)
            else:
                self._say(f"{at}: the program is not a mapping or a list of steps")
        return tuple(runs)

    def _step(self, at, st, decl, roles):
        if not isinstance(st, dict):
            self._say(f"{at}: a step is not a mapping")
            return
        for k in st:
            if k not in STEP_KEYS:
                self._say(f"{at}: unknown step key {k!r}")
        if "limb" not in st or (st["limb"] is not None and st["limb"] not in decl["limbs"]):
            self._say(f"{at}: no such limb {st.get('limb')!r}")
        if st.get("until") not in decl["step_tests"]:
            self._say(f"{at}: unknown step test {st.get('until')!r}")
        m = st.get("match")
        if (st.get("limb") is None) != (m is None):
            self._say(f"{at}: a step that asks a limb matches a row, and only such a step")
        elif m is not None and m not in STEP_MATCH and not (
                isinstance(m, dict) and set(m) == {"tag"} and isinstance(m["tag"], str)):
            self._say(f"{at}: match {m!r} is not one of {STEP_MATCH} or {{tag: <tag>}}")
        if "aim" in st and (not isinstance(st["aim"], str) or st["aim"] not in roles):
            self._say(f"{at}: aim {st['aim']!r} is not a role")
        if not isinstance(st.get("flee", False), bool):
            self._say(f"{at}: flee is not true or false")

    def _durations(self, where, spec, runs):
        if spec is None:
            spec = {}
        if not isinstance(spec, dict):
            self._say(f"{where}: duration is not a mapping")
            return ()
        for be in spec:
            if be not in self.backends:
                self._say(f"{where}: duration for unknown backend {be!r}")
        out = []
        for be in runs:
            d = spec.get(be)
            if d is None:
                self._say(f"{where}: it runs on {be}, but it has no {be} duration")
            elif d != "route" and (isinstance(d, bool) or not isinstance(d, int) or d < 1):
                self._say(f"{where}: the {be} duration is not a positive integer or route: {d!r}")
            else:
                out.append((be, d))
        return tuple(out)

    def _templates(self, where, v, roles):
        """Check the words. `text` uses every role that is not optional and no
        optional one; `text_with` uses every role; the told reports name the
        actor. Returns the templates that are present."""
        need = {r.name for r in roles if not r.optional}
        opt = {r.name for r in roles if r.optional}
        out = {}
        for key in TEMPLATES:
            tpl = v.get(key)
            if tpl is None:
                continue
            used = _fields(tpl) if isinstance(tpl, str) and tpl.strip() else None
            if used is None:
                self._say(f"{where}: {key} is not a plain template: {tpl!r}")
                continue
            out[key] = tpl
            allowed = need | opt | ({"actor"} if key.startswith("told") else set())
            if used - allowed:
                self._say(f"{where}: {key} uses {sorted(used - allowed)}, which are not roles")
            if key.startswith("told") and "actor" not in used:
                self._say(f"{where}: {key} does not name the actor")
            if key == "text" and (need - used or used & opt):
                self._say(f"{where}: text uses every role that is not optional, and no optional role")
            if key == "text_with" and (need | opt) - used:
                self._say(f"{where}: text_with uses every role")
            if key in ("told", "told_start") and used & opt:
                self._say(f"{where}: {key} uses an optional role, which can be empty")
            if key == "told_with" and not used & opt:
                self._say(f"{where}: told_with uses no optional role")
        if "text" not in out and v.get("text") is None:
            self._say(f"{where}: no text")
        if opt and "text_with" not in out:
            self._say(f"{where}: an optional role needs text_with")
        if not opt and ("text_with" in out or "told_with" in out):
            self._say(f"{where}: text_with and told_with are for a verb with an optional role")
        return out

    def _whole(self, verbs):
        nulls = [k for k, v in verbs.items() if v.null_act]
        if len(nulls) != 1:
            self._say(f"exactly one verb is the null act, not {nulls}")
        for be, spec in self._raw["backends"].items():
            for native in spec.get("verbs", ()):
                if not any(self._programs[k].get(be) == {"verb": native} for k in verbs):
                    self._say(f"backend {be!r}: no verb runs its verb {native!r}")

    # ------------------------------------------------------------ reading
    def order(self, verb_id):
        """The place of a verb in canonical menu order (group, then file order)."""
        return self._rank[verb_id]

    def runnable(self, backend):
        """The ids of the verbs that `backend` runs, in canonical order."""
        return tuple(sorted((k for k, v in self.verbs.items() if backend in v.runs_on),
                            key=self.order))

    def program(self, verb_id, backend):
        """A new copy of the program of a verb on a backend, or None."""
        return copy.deepcopy(self._programs[verb_id].get(backend))

    def cap_test(self, cap, backend):
        """A new copy of the test of a capability on a backend, or None."""
        return copy.deepcopy(self._raw["capabilities"][cap].get(backend))

    def drive_test(self, drive, backend):
        """A new copy of the test of a drive on a backend, or None."""
        return copy.deepcopy(self._raw["drives"][drive].get(backend))

    def declared(self, backend):
        """A new copy of what a backend declares that it can run."""
        return copy.deepcopy(self._raw["backends"][backend])

    def option(self, backend, verb_id, roles, duration=None, features=(), missing=()):
        """An Option of a verb with its roles filled (`roles` maps each role to
        a Ref). The catalog gives the words, the group, and the duration; give
        `duration` for a verb whose duration is "route". An option with
        `missing` capability words is blocked."""
        verb = self.verbs[verb_id]
        if backend not in verb.runs_on:
            raise ValueError(f"verb {verb_id} does not run on {backend}")
        for name, ref in roles.items():
            r = verb.role(name)
            if r is None:
                raise ValueError(f"verb {verb_id} has no role {name!r}")
            if ref.kind != r.kind:
                raise ValueError(f"verb {verb_id}: role {name!r} takes a {r.kind}, not a {ref.kind}")
        for r in verb.roles:
            if not r.optional and r.name not in roles:
                raise ValueError(f"verb {verb_id}: role {r.name!r} is not filled")
        if duration is None:
            duration = verb.duration(backend)
            if duration == "route":
                raise ValueError(f"verb {verb_id}: give the duration of the route on {backend}")
        return Option(verb_id, dict(roles), verb.option_text({k: r.name for k, r in roles.items()}),
                      verb.group, duration, bool(verb.interrupt_on), features,
                      ok=not missing, missing=tuple(missing))
