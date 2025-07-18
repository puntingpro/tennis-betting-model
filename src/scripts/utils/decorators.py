from functools import wraps
from typing import Any, Callable

from .logger import setup_logging


def with_logging(func: Callable) -> Callable:
    """
    A decorator that sets up logging before running a function.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Assumes the first argument is the args namespace from argparse
        if args and hasattr(args[0], "verbose"):
            setup_logging(
                level="DEBUG" if args[0].verbose else "INFO",
                json_logs=getattr(args[0], "json_logs", False),
            )
        else:
            setup_logging()
        return func(*args, **kwargs)

    return wrapper
