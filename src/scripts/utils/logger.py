import logging
import sys

from pythonjsonlogger import jsonlogger


def setup_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """
    Configure the root logger for the application.
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Use a basic StreamHandler for regular logs
    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        formatter = jsonlogger.JsonFormatter(log_format)
    else:
        formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)

    # Configure the root logger
    logging.basicConfig(level=log_level, handlers=[handler], force=True)


# --- Reusable Logging Functions ---

def log_info(msg: str, *args: object) -> None:
    logging.info(msg, *args)


def log_warning(msg: str, *args: object) -> None:
    logging.warning(f"⚠️ {msg}", *args)


def log_error(msg: str, *args: object) -> None:
    logging.error(f"❌ {msg}", *args)


def log_success(msg: str, *args: object) -> None:
    logging.info(f"✅ {msg}", *args)