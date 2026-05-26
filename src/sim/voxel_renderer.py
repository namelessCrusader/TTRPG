"""Deprecated: use ``python3 -m src.sim.play --pygame`` instead."""

from __future__ import annotations

import sys
import warnings


def main(argv=None) -> None:
    warnings.warn(
        "voxel_renderer merged into src.sim.play — use: "
        "python3 -m src.sim.play --pygame [--gl] --world WORLD",
        DeprecationWarning,
        stacklevel=2,
    )
    from .play import main as play_main

    if argv is None:
        argv = sys.argv[1:]
    play_main(["--pygame", *argv])


if __name__ == "__main__":
    main()
