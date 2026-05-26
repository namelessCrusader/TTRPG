"""Mirror stdout/stderr to a log file (for long autonomous + debug runs)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO


class TeeStream:
    """Write to terminal and a log file."""

    def __init__(self, terminal: TextIO, log_file: TextIO):
        self._terminal = terminal
        self._log = log_file

    def write(self, data: str) -> int:
        self._terminal.write(data)
        self._log.write(data)
        self._log.flush()
        return len(data)

    def flush(self) -> None:
        self._terminal.flush()
        self._log.flush()

    def isatty(self) -> bool:
        return self._terminal.isatty()


class LogTeeSession:
    """Context manager: restore stdout/stderr on exit."""

    def __init__(self, path: Path):
        self.path = path
        self._fh: Optional[TextIO] = None
        self._stdout = None
        self._stderr = None

    def __enter__(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")
        self._fh.write(
            f"# Mark_1 simulation log — {datetime.now().isoformat(timespec='seconds')}\n"
        )
        self._fh.flush()
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = TeeStream(self._stdout, self._fh)  # type: ignore[assignment]
        sys.stderr = TeeStream(self._stderr, self._fh)  # type: ignore[assignment]
        return self.path

    def __exit__(self, *args) -> None:
        if self._stdout is not None:
            sys.stdout = self._stdout
        if self._stderr is not None:
            sys.stderr = self._stderr
        if self._fh is not None:
            self._fh.close()


def default_log_path(prefix: str = "autonomous") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("logs") / f"{prefix}_{ts}.log"
