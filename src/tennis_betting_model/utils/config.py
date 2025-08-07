# src/tennis_betting_model/utils/config.py
from pathlib import Path
from typing import Any, Dict
import yaml
from pydantic import ValidationError
import os
from functools import lru_cache

from .logger import log_error, log_info
from .config_schema import Config


def _merge_configs(base: dict, override: dict) -> dict:
    """Recursively merge override config into base config."""
    for key, value in override.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            base[key] = _merge_configs(base[key], value)
        else:
            base[key] = value
    return base


@lru_cache(maxsize=None)
def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a base YAML configuration and merges an environment-specific
    override file if it exists, determined by the APP_ENV variable.
    The result is cached to avoid repeated file I/O.
    """
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Base config file not found at: {config_path}")

    with p.open("r") as f:
        config_dict = dict(yaml.safe_load(f))

    # Check for an environment-specific config override file
    env = os.getenv("APP_ENV", "dev")
    env_config_path = p.parent / f"{p.stem}.{env}.yaml"

    if env_config_path.exists():
        log_info(f"Found environment config, loading overrides from: {env_config_path}")
        with env_config_path.open("r") as f:
            env_config_dict = dict(yaml.safe_load(f))
            config_dict = _merge_configs(config_dict, env_config_dict)
    else:
        log_info(f"No environment config for '{env}' found. Using base config only.")

    try:
        # Validate the final merged dictionary against the Pydantic model
        Config(**config_dict)
        log_info("✅ Configuration file validation successful.")
    except ValidationError as e:
        log_error(
            f"❌ Configuration file '{p.name}' (with overrides for env '{env}') is invalid!"
        )
        log_error(str(e))
        raise

    return config_dict
