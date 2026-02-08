from __future__ import annotations

import sys
from pathlib import Path


def _add_src_to_path() -> None:
    src = Path(__file__).resolve().parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> int:
    _add_src_to_path()
    from ai_news_dashboard.pipeline import main as pipeline_main

    return pipeline_main()


if __name__ == "__main__":
    raise SystemExit(main())
