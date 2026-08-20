"""logging_config.py — structured logging setup"""
import logging
import sys
from pathlib import Path


def setup_logging(log_dir: str = "logs", level: str = "INFO") -> None:
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s  %(levelname)-8s  %(name)-25s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir_path / "pipeline.log", mode="a"),
    ]

    logging.basicConfig(
        level   = getattr(logging, level.upper(), logging.INFO),
        format  = fmt,
        datefmt = datefmt,
        handlers= handlers,
        force   = True,
    )

    # Suppress noisy third-party loggers
    for noisy in ["urllib3", "requests", "numba", "torch", "esm", "h5py"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
