from pathlib import Path
from typing import Any, Dict
import yaml
from pydantic import ValidationError

from .logger import log_error, log_info
from .config_schema import Config


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file from the specified path and validates it.
    Args:
        config_path (str): The path to the YAML config file.
    Returns:
        Dict[str, Any]: A dictionary representing the configuration.
    Raises:
        FileNotFoundError: If the config file does not exist at the given path.
        ValidationError: If the config file does not match the defined schema.
    """
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with p.open("r") as f:
        config_dict = dict(yaml.safe_load(f))

    try:
        # Validate the dictionary against the Pydantic model
        Config(**config_dict)
        log_info("✅ Configuration file validation successful.")
    except ValidationError as e:
        log_error(f"❌ Configuration file '{config_path}' is invalid!")
        # FIX: Convert validation error to string to resolve mypy arg-type error
        log_error(str(e))
        raise

    return config_dict
