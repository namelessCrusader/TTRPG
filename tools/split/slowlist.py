"""Turn one measured run into the `slow` marker list.

    pytest tests/test_voxel.py tests/test_wounds.py tests/test_sound.py \
        tests/test_projectiles.py tests/test_mpm.py -p no:cacheprovider \
        --durations=0 -q > durations.txt
    python3 slowlist.py durations.txt [budget_seconds=90]

Marks the SMALLEST set of tests whose removal brings the rest under the budget
(serial seconds), slowest first, and prints them as the `SLOW = {...}` set
for tests/conftest.py ("file::test" keys). Setup/teardown time is charged to
the test. Parametrised ids collapse onto their function: a slow case marks
the whole function.
"""
import collections
import re
import sys

path = sys.argv[1]
budget = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0
cost = collections.Counter()
for line in open(path):
    m = re.match(r"\s*([\d.]+)s\s+(call|setup|teardown)\s+(\S+)", line)
    if m:
        cost[m.group(3)] += float(m.group(1))
total = sum(cost.values())
slow, rest = [], total
for node, secs in cost.most_common():
    if rest <= budget:
        break
    slow.append((node, secs))
    rest -= secs
print(f"{len(cost)} tests, {total:.0f}s serial; marking {len(slow)} slow "
      f"({total - rest:.0f}s) leaves {rest:.0f}s in the fast tier")
print("SLOW = {")
for node, secs in slow:
    f, _, name = node.partition("::")
    key = f.split("/")[-1] + "::" + name.split("[")[0]
    print(f"    {key!r},{' ' * max(1, 72 - len(key))}# {secs:5.1f}s")
print("}")
