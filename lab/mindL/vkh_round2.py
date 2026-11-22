"""Round-2 lifetimes again, to read the learned worth of "got_kind" (vk) and "got_harm" (vh) (library need 2, S8.1).

Round 2 did not save them. This runs L12 lives A and B of the fixed mind (mind2, flags all2, the round-2 settings in
lab/results/mindL2/settings.json) for seeds 1-3 to night 400 (the night the partners turn deaf in the 600-night
lives), logs vk and vh at every night (tests_L.night_row "vkh"), and checks that the first 400 nights of
drive error equal the stored round-2 lives.

    MINDL_TUNED=lab/results/mindL2/settings.json MINDL_RUNS=<dir> python -m lab.mindL.vkh_round2 <dir>
"""
import json
import sys
from pathlib import Path

from lab.mindL import tests_L as T

OLD = Path(__file__).resolve().parents[1] / "results" / "mindL2" / "runs"


def main(out):
    T.RUNS = Path(out)
    todo = [s for s in T.specs("L12", 600, ("mind2",), "all2") if s["variant"] in ("A", "B") and s["seed"] in (1, 2, 3)]
    for s in todo:
        old_id = s["id"]
        s["nights"], s["id"] = 400, old_id.replace("_n600_", "_n400_")      # same partner phases (deaf from 400)
        print(T.run(s), flush=True)
        a = json.loads((T.RUNS / f"{s['id']}.json").read_text())["drive_error"]
        b = json.loads((OLD / f"{old_id}.json").read_text())["drive_error"][:400]
        print(old_id, "first 400 nights equal the stored round-2 life:", a == b, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
