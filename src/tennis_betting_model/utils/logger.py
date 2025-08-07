import logging
import sys


def setup_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure the root logger for the application."""

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(log_format))
    handler.stream.reconfigure(encoding="utf-8")  # type: ignore

    logging.basicConfig(level=log_level, handlers=[handler], force=True)


def log_info(msg: str) -> None:
    logging.info(msg)


def log_warning(msg: str) -> None:
    logging.warning(f"WARNING: {msg}")


def log_error(msg: str) -> None:
    logging.error(f"ERROR: {msg}")


def log_success(msg: str) -> None:
    logging.info(f"[SUCCESS] {msg}")
