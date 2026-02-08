from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logger(name: str = "her2st", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Create or fetch a configured logger with stream and optional file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def log_step(logger: logging.Logger, message: str) -> None:
    logger.info("[STEP] %s", message)


def log_stat(logger: logging.Logger, key: str, value: object) -> None:
    logger.info("[STAT] %s=%s", key, value)


def get_default_logger() -> logging.Logger:
    return setup_logger()


if __name__ == "__main__":
    log = setup_logger("her2st.utils")
    log_step(log, "utils module initialized")
