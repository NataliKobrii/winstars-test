"""Logging setup used across the project."""

import logging

from config import DEBUG, LOG_DIR, LOG_FILE, LOG_LEVEL


def format_error_message(base_message: str) -> str:
    """Append log-file path to an error message when DEBUG is enabled.

    Parameters:
        base_message: str
            User-facing error message.

    Returns:
        str:
            Error message, extended with log-file information in debug mode.
    """
    if DEBUG:
        return f"{base_message} See {LOG_FILE} for full traceback."
    return base_message


def setup_logging(logger_name: str = "task2") -> logging.Logger:
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
