from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML config file.
    """
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with p.open("r") as f:
        return yaml.safe_load(f)