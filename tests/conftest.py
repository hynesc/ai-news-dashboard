from __future__ import annotations

from pathlib import Path
import sys


def pytest_configure() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

