"""
main.py — Sentinel Desktop AI Agent
────────────────────────────────────
Entry point.  Run with:

    python main.py

Optional flags
    --checkpoint  PATH   Override checkpoint directory  (default: checkpoints/)
    --db          PATH   Override SQLite database path  (default: data/sentinel.db)
    --debug              Enable verbose logging
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# ── ensure project root is on the path ────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── suppress noisy third-party loggers before anything imports them ────────
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("comtypes").setLevel(logging.WARNING)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore    import Qt
from PyQt5.QtGui     import QIcon, QFont, QFontDatabase

from core.agent.agent import SentinelAgent
from ui.window         import SentinelWindow


# ──────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sentinel Desktop AI Agent")
    p.add_argument("--checkpoint", default="checkpoints",
                   help="Directory containing model checkpoint files")
    p.add_argument("--db", default="data/sentinel.db",
                   help="SQLite database path")
    p.add_argument("--debug", action="store_true",
                   help="Enable verbose logging")
    return p.parse_args()


def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    fmt   = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    logging.basicConfig(level=level, format=fmt)


def ensure_directories() -> None:
    for d in ("checkpoints", "data", Path.home() / "Pictures" / "Sentinel Screenshots"):
        Path(d).mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    setup_logging(args.debug)
    ensure_directories()

    log = logging.getLogger("sentinel.main")
    log.info("Starting Sentinel…")

    # ── Qt application ────────────────────────────────────────────────
    # High-DPI support (must be set before QApplication is created)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps,    True)

    app = QApplication(sys.argv)
    app.setApplicationName("Sentinel")
    app.setApplicationDisplayName("Sentinel AI")
    app.setOrganizationName("Sentinel")

    # ── Agent ─────────────────────────────────────────────────────────
    log.info("Initialising agent (checkpoint=%s, db=%s)…",
             args.checkpoint, args.db)
    try:
        agent = SentinelAgent(
            checkpoint_dir=args.checkpoint,
            db_path=args.db,
        )
    except Exception:
        log.exception("Failed to initialise agent")
        return 1

    # ── Main window ───────────────────────────────────────────────────
    window = SentinelWindow(agent=agent)
    window.show()

    log.info("Sentinel is running.")
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
