"""Central logging configuration for advanced_rag."""
import logging
import sys
from pathlib import Path

_LOG_FILE = Path(__file__).resolve().parent / "rag.log"


def setup(level: int = logging.DEBUG) -> None:
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    if root.handlers:
        return  # already configured
    root.setLevel(level)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    root.addHandler(sh)
    root.addHandler(fh)
