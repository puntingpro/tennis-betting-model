from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file from the specified path.

    Args:
        config_path (str): The path to the YAML config file.

    Returns:
        Dict[str, Any]: A dictionary representing the configuration.

    Raises:
        FileNotFoundError: If the config file does not exist at the given path.
    """
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with p.open("r") as f:
        return dict(yaml.safe_load(f))
