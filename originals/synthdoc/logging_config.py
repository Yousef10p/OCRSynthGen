"""
Centralized logging configuration for synthetic document generation.

Provides configurable logging that library users can control.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability."""

    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        """Format log record with colors."""
        # Save original levelname
        orig_levelname = record.levelname

        # Add color if terminal supports it
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            if record.levelname in self.COLORS:
                record.levelname = (
                    f"{self.COLORS[record.levelname]}"
                    f"{record.levelname}{self.RESET}"
                )

        result = super().format(record)

        # Restore original levelname
        record.levelname = orig_levelname

        return result


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
    logger_name: Optional[str] = None
) -> logging.Logger:
    """
    Setup centralized logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: ./logs)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        logger_name: Name of logger to configure (None = root logger)

    Returns:
        Configured logger

    Example:
        >>> from synthdoc.logging_config import setup_logging
        >>> logger = setup_logging(log_level="DEBUG")
        >>> logger.info("Generation started")
    """
    # Get logger
    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger()

    # Set level
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            '%(levelname)s - %(name)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        if log_dir is None:
            log_dir = Path("logs")

        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"generation_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    # Prevent propagation to avoid duplicate logs
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance

    Example:
        >>> from synthdoc.logging_config import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.debug("Debug message")
    """
    return logging.getLogger(name)


def disable_logging():
    """Disable all logging (useful for library users who don't want logs)."""
    logging.disable(logging.CRITICAL)


def enable_logging():
    """Re-enable logging after it was disabled."""
    logging.disable(logging.NOTSET)


# Module-level convenience function
_default_logger_configured = False


def configure_default_logger(log_level: str = "INFO") -> logging.Logger:
    """
    Configure default logger for the library.

    This is called automatically on first import if not already configured.
    Library users can call setup_logging() for more control.

    Args:
        log_level: Logging level

    Returns:
        Configured logger
    """
    global _default_logger_configured

    if not _default_logger_configured:
        logger = setup_logging(
            log_level=log_level,
            log_to_file=False,  # Default: console only
            log_to_console=True
        )
        _default_logger_configured = True
        return logger

    return logging.getLogger()


# Auto-configure on import (console only, INFO level)
# Users can override by calling setup_logging() themselves
if not _default_logger_configured:
    configure_default_logger("INFO")
