import logging
import sys


def setup_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure the root logger for the application."""

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_level = getattr(logging, level.upper(), logging.INFO)

    # --- FIXED: Explicitly set the stream to use UTF-8 encoding ---
    # This handler ensures all output, including piped output, uses UTF-8.
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(log_format))

    # Force the handler to use UTF-8 encoding to prevent UnicodeEncodeError on Windows
    # when piping output to a file.
    handler.stream.reconfigure(encoding="utf-8")  # type: ignore

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
