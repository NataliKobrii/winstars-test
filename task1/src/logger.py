"""Logging setup used across the project."""

import logging

from config import LOG_DIR, LOG_FILE, LOG_LEVEL


def setup_logging(logger_name: str = "task1") -> logging.Logger:
    """Create and return a configured logger.

    Parameters:
        logger_name: str
            Name of the logger.

    Returns:
        logging.Logger:
            Configured logger instance.
    """
    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.ERROR))

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # File handler for persistent logs
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )

    logger.addHandler(file_handler)

    # Disable propagation to prevent duplicate logs in root logger
    logger.propagate = False

    return logger
