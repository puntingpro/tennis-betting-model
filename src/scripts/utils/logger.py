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

def log_info(msg: str) -> None:
    logging.info(msg)


def log_warning(msg: str) -> None:
    logging.warning(f"⚠️ {msg}")


def log_error(msg: str) -> None:
    logging.error(f"❌ {msg}")


def log_success(msg: str) -> None:
    logging.info(f"✅ {msg}")