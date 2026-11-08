"""WAVE 13 PHASE A AUDIT (lane A-X).

    python3 tools/w13_audit.py [--base DIR] [--cross] [--fast] [--no-import]

Checks, prints a table, and exits 1 on a hard failure:
  1. no existing file differs from base18 (a sandbox file is allowed to differ
     only by being NEW); lists modified and deleted files
  2. no two lanes declare the same file, and no new basename is used twice
  3. every file in the sandbox that is not in base18 is in the manifest
  4. every new src/voxel module imports ALONE, without pulling in World
     (src.voxel.sim); each import runs in one process with the package modules
     purged between imports
  5. a table: module, owner topic, lane, lines, imports alone
  6. tag counts ([I] [U] [E] [C]) of the tests of each lane
  7. the GLOB TIER (on unless --no-glob): the existing tests that list the source tree (a size
     budget, a hygiene scan, "only these modules read X") run in a base18 copy and in the sandbox.
     A new file can break an old test only that way. A failure listed in KNOWN_PHASE_B (a phase B
     edit of an existing test) is reported as PHASE-B FIX and does not fail the audit; any other
     failure does, and so does a KNOWN_PHASE_B entry that now passes (stale).
With --cross it also runs tests/test_w13_cross.py and lists every skipped case
with the reason (the missing module). With --fast it runs the fast tier of the
EXISTING tests (those that are in base18) in the sandbox and in a copy of base18
and compares the pass counts (heavy: take the test lock and cap.sh yourself, or
let this script do it: it wraps the run in flock and cap.sh when they exist).

Standard library only; never edits a file.
"""
import argparse
import ast
import filecmp
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import w13_manifest as M   # noqa: E402

MARK = os.environ.get("MARK1_BUILD", "/home/nisargparikh/.cache/mark1-build")
DEFAULT_BASE = os.path.join(MARK, "base18")
LOCK = os.path.join(MARK, "test.lock")
CAP = os.path.join(MARK, "cap.sh")
IGNORE_DIRS = {"__pycache__", ".pytest_cache", ".git", ".mypy_cache"}
IGNORE_SUFFIX = (".pyc", ".pyo")


# ---- 1, 3: compare the trees ------------------------------------------------------
def tree(root):
    out = {}
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in IGNORE_DIRS]
        for f in fns:
            if f.endswith(IGNORE_SUFFIX):
                continue
            out[os.path.relpath(os.path.join(dp, f), root)] = os.path.join(dp, f)
    return out


def compare(sandbox, base):
    """(modified, new, deleted): relative paths."""
    a, b = tree(sandbox), tree(base)
    modified = sorted(p for p in a if p in b and not filecmp.cmp(a[p], b[p], shallow=False))
    new = sorted(p for p in a if p not in b)
    deleted = sorted(p for p in b if p not in a)
    return modified, new, deleted


# ---- 2: duplicate declarations ------------------------------------------------------
def duplicates():
    """Paths declared by two lanes, and basenames used by two different paths."""
    seen, dup_paths = {}, []
    for lane, topic, _batch, files, _tests in M._LANES:
        for f in files:
            if f in seen and seen[f] != lane:
                dup_paths.append((f, seen[f], lane))
            seen.setdefault(f, lane)
    by_base = {}
    for f, lane in seen.items():
        by_base.setdefault(os.path.basename(f), []).append((f, lane))
    dup_names = [(b, v) for b, v in sorted(by_base.items()) if len(v) > 1]
    return dup_paths, dup_names


# ---- 4: import alone ------------------------------------------------------------------
_IMPORT_PROBE = r"""
import importlib, json, sys, traceback
sys.path.insert(0, %(root)r)
out = {}
for name in %(mods)r:
    for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
        del sys.modules[k]
    try:
        importlib.import_module("src.voxel." + name)
    except BaseException as e:
        out[name] = "ERR " + type(e).__name__ + ": " + str(e).splitlines()[0][:150] if str(e) else "ERR " + type(e).__name__
        continue
    pulled = sorted(k for k in sys.modules if k in ("src.voxel.sim", "src.voxel.world"))
    out[name] = ("PULLS " + ",".join(pulled)) if pulled else "ok"
print("JSON" + json.dumps(out))
"""


def import_alone(sandbox, names):
    """{module: 'ok' | 'ERR ...' | 'PULLS ...'}; one process, package purged between imports."""
    code = _IMPORT_PROBE % {"root": sandbox, "mods": list(names)}
    r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True,
                       cwd=sandbox, timeout=300)
    for line in r.stdout.splitlines():
        if line.startswith("JSON"):
            return json.loads(line[4:])
    return {n: "ERR probe failed: " + (r.stderr.strip().splitlines() or ["?"])[-1] for n in names}


# ---- 6: tag counts ----------------------------------------------------------------------
def test_functions(path):
    """Names of the test functions in a test file (module level and in classes)."""
    try:
        tree_ = ast.parse(open(path).read())
    except (OSError, SyntaxError):
        return []
    names = []
    for n in ast.walk(tree_):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test"):
            names.append(n.name)
    return names


def tag_counts(sandbox):
    """{lane: {tag: n, 'files': k, 'missing': [...]}} from the lane's test files."""
    out = {}
    for lane, d in M.LANES.items():
        c = {t: 0 for t in M.TAGS}
        missing = []
        for t in d["tests"]:
            p = os.path.join(sandbox, t)
            if not os.path.exists(p):
                missing.append(t)
                continue
            for name in test_functions(p):
                c[M.tag_of(name)] += 1
        c["missing"] = missing
        out[lane] = c
    return out


# ---- cross tests and the fast tier ----------------------------------------------------
def wrap(cmd, heavy):
    pre = []
    if heavy and os.path.exists(LOCK):
        pre += ["flock", "-w", "3000", LOCK]
    if os.path.exists(CAP):
        pre += [CAP]
    return pre + cmd


def run_cross(sandbox):
    cmd = wrap([sys.executable, "-B", "-m", "pytest", "-q", "-rsx", "-m", "not slow", "-p", "no:cacheprovider",
                "tests/test_w13_cross.py"], heavy=False)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=sandbox)
    skips = [ln for ln in r.stdout.splitlines() if ln.startswith("SKIPPED")]
    xfails = [ln for ln in r.stdout.splitlines() if ln.startswith("XFAIL")]
    fails = [ln for ln in r.stdout.splitlines() if ln.startswith("FAILED")]
    return r.returncode, skips, xfails, fails, r.stdout.strip().splitlines()[-1:] or [""]


_SUMMARY = re.compile(r"(\d+) (passed|failed|error|errors|skipped|deselected)")


def parse_summary(text):
    got = {}
    for n, k in _SUMMARY.findall(text.strip().splitlines()[-1] if text.strip() else ""):
        got["errors" if k.startswith("error") else k] = int(n)
    return got


def existing_test_files(base):
    return sorted(os.path.join("tests", f) for f in os.listdir(os.path.join(base, "tests"))
                  if f.startswith("test_") and f.endswith(".py"))


def run_fast(tree_root, files):
    cmd = wrap([sys.executable, "-B", "-m", "pytest", "-q", "-m", "not slow",
                "-p", "no:cacheprovider"] + files, heavy=True)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=tree_root)
    return parse_summary(r.stdout), r.stdout.strip().splitlines()[-3:]


def run_ids(tree_root, ids):
    """(set of failed node ids, summary line) of the given existing tests run in tree_root."""
    cmd = wrap([sys.executable, "-B", "-m", "pytest", "-q", "-rfE", "-m", "not slow",
                "-p", "no:cacheprovider"] + ids, heavy=False)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=tree_root, env=env)
    failed = {re.split(r"\s+-\s+", ln.split(None, 1)[1])[0].strip()
              for ln in r.stdout.splitlines() if ln.startswith(("FAILED", "ERROR")) and len(ln.split(None, 1)) > 1}
    return failed, (r.stdout.strip().splitlines() or [""])[-1]


def glob_tier(sandbox, base, base_copy):
    """Run the source-globbing existing tests in the sandbox and in the base18 copy.
    Returns (ids, failed_in_sandbox, failed_in_base, summaries)."""
    ids = M.source_globbing_tests(base)
    mine, s_mine = run_ids(sandbox, ids)
    theirs, s_base = run_ids(base_copy if os.path.isdir(base_copy) else base, ids)
    return ids, mine, theirs, (s_mine, s_base)


# ---- main -----------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--sandbox", default=ROOT)
    ap.add_argument("--cross", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--base-copy", default=None, help="a scratch copy of base18 to run the baseline fast tier in")
    ap.add_argument("--no-import", action="store_true")
    ap.add_argument("--no-glob", action="store_true", help="skip the glob tier (the existing tests that list src/)")
    ap.add_argument("--live", action="store_true", help="with --fast: run both sides now instead of reading w13/A-X/*_fast.log")
    a = ap.parse_args(argv)
    bad = 0

    modified, new, deleted = compare(a.sandbox, a.base)
    print(f"existing files modified vs base18: {len(modified)}")
    for p in modified:
        print("   MODIFIED", p)
    print(f"existing files deleted vs base18:  {len(deleted)}")
    for p in deleted:
        print("   DELETED", p)
    bad += len(modified) + len(deleted)

    dup_paths, dup_names = duplicates()
    print(f"duplicate new paths (two lanes):   {len(dup_paths)}")
    for p in dup_paths:
        print("   DUP", p)
    print(f"duplicate new basenames:           {len(dup_names)}")
    for b, v in dup_names:
        print("   DUPNAME", b, v)
    bad += len(dup_paths) + len(dup_names)

    declared = set(M.new_files())
    unlisted = [p for p in new if p not in declared]
    print(f"new files not in the manifest:     {len(unlisted)}")
    for p in unlisted:
        print("   UNLISTED", p)
    bad += len(unlisted)
    built = [p for p in M.new_files() if os.path.exists(os.path.join(a.sandbox, p))]
    print(f"manifest files present:            {len(built)} of {len(declared)}")

    mods = [m for m in M.modules() if os.path.exists(os.path.join(a.sandbox, "src/voxel", m + ".py"))]
    alone = {} if a.no_import else import_alone(a.sandbox, mods)
    warned = []
    print()
    print(f"{'module':<16}{'topic':<6}{'lane':<8}{'lines':>6}  imports alone")
    for m in M.modules():
        p = os.path.join(a.sandbox, "src/voxel", m + ".py")
        lane = M.FILE_LANE["src/voxel/" + m + ".py"]
        if os.path.exists(p):
            n = sum(1 for _ in open(p))
            state = alone.get(m, "not run")
        else:
            n, state = 0, "ABSENT"
        if state.startswith("PULLS") and m.startswith(M.SCENE_PREFIX):
            state = "scene (builds a World by design: " + state + ")"
            warned.append(m)
        print(f"{m:<16}{M.MODULES[m]:<6}{lane:<8}{n:>6}  {state}")
        if state.startswith(("ERR", "PULLS")):
            bad += 1
    print(f"modules present: {len(mods)} of {len(M.MODULES)}; scene modules that import World: {warned}")

    print()
    tc = tag_counts(a.sandbox)
    print(f"{'lane':<8}{'[I]':>5}{'[U]':>5}{'[E]':>5}{'[C]':>5}   tests missing")
    tot = {t: 0 for t in M.TAGS}
    for lane, c in tc.items():
        print(f"{lane:<8}{c['I']:>5}{c['U']:>5}{c['E']:>5}{c['C']:>5}   {', '.join(c['missing'])}")
        for t in M.TAGS:
            tot[t] += c[t]
    print(f"{'total':<8}{tot['I']:>5}{tot['U']:>5}{tot['E']:>5}{tot['C']:>5}")

    pending = 0
    if not a.no_glob:
        work = os.path.join(MARK, "w13", "A-X")
        ids, mine, theirs, (s_mine, s_base) = glob_tier(a.sandbox, a.base, a.base_copy or os.path.join(work, "b18"))
        print(f"\nglob tier: {len(ids)} existing tests that list the source tree")
        print(f"   sandbox: {s_mine}")
        print(f"   base18:  {s_base}")
        fixes = sorted(f for f in mine if f in M.KNOWN_PHASE_B)
        news = sorted(f for f in mine if f not in M.KNOWN_PHASE_B and f not in theirs)
        stale = sorted(f for f in M.KNOWN_PHASE_B if f in ids and f not in mine)
        for f in fixes:
            print("   PHASE-B FIX  ", f, "\n                ", M.KNOWN_PHASE_B[f])
        for f in news:
            print("   BROKEN BY A NEW FILE", f)
        for f in sorted(theirs):
            print("   FAILS IN BASE18 TOO", f)
        for f in stale:
            print("   STALE KNOWN_PHASE_B (now passes, remove the entry)", f)
        bad += len(news) + len(stale)
        pending = len(fixes)

    if a.cross:
        rc, skips, xfails, fails, last = run_cross(a.sandbox)
        print(f"\ncross tests: exit {rc}; {last[0]}")
        print(f"skipped cross tests (module or name missing): {len(skips)}")
        for s in skips:
            print("  ", s)
        print(f"known differences reported by xfail (both values): {len(xfails)}")
        for s in xfails:
            print("  ", s[:400])
        for s in fails:
            print("  ", s)
        bad += 1 if rc else 0

    if a.fast:
        files = existing_test_files(a.base)
        work = os.path.join(MARK, "w13", "A-X")
        mine = parse_summary(open(os.path.join(work, "sandbox_fast.log")).read()) \
            if os.path.exists(os.path.join(work, "sandbox_fast.log")) and not a.live else None
        if not mine:
            mine, mtail = run_fast(a.sandbox, files)
        print(f"\nfast tier, sandbox, {len(files)} existing test files: {mine}")
        base_copy = a.base_copy or os.path.join(work, "b18")
        theirs = parse_summary(open(os.path.join(work, "base18_fast.log")).read()) \
            if os.path.exists(os.path.join(work, "base18_fast.log")) and not a.live else None
        if not theirs and os.path.isdir(base_copy):
            theirs, btail = run_fast(base_copy, files)
        if theirs:
            print(f"fast tier, base18 copy: {theirs}")
            same = (mine.get("passed") == theirs.get("passed") and mine.get("failed", 0) == theirs.get("failed", 0)
                    and mine.get("errors", 0) == theirs.get("errors", 0))
            print("fast tier unchanged:", "YES" if same else "NO")
            bad += 0 if same else 1
    print("\nAUDIT", "FAIL" if bad else "OK", f"({bad} problems; {pending} phase B fix(es) of existing tests pending)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
