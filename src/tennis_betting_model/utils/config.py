# src/tennis_betting_model/utils/config.py
from typing import Any, Dict, cast
from pydantic import ValidationError
from omegaconf import DictConfig, OmegaConf

from .logger import log_error, log_info
from .config_schema import Config


def validate_config(config: DictConfig) -> Dict[str, Any]:
    """
    Validates an OmegaConf DictConfig object against the Pydantic model.
    """
    try:
        # Convert OmegaConf to a standard Python dict for Pydantic validation
        config_dict = OmegaConf.to_container(config, resolve=True)

        # Ensure config_dict is a mapping before unpacking to satisfy mypy
        assert isinstance(
            config_dict, dict
        ), f"Config validation failed: expected a dict, but got {type(config_dict)}"

        # Pass the validated dict to the Pydantic model
        Config(**config_dict)
        log_info("✅ Configuration validation successful.")
        return cast(Dict[str, Any], config_dict)
    except ValidationError as e:
        log_error("❌ Configuration is invalid!")
        log_error(str(e))
        raise
