from __future__ import annotations

import logging


def setup_logging(*, level: int = logging.INFO) -> None:
    # Avoid double-configuring if called multiple times (Streamlit reloads, tests, etc.)
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
