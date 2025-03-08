"""
Settings manager for DepenSage.

This module handles loading configuration settings using a module-level singleton pattern.
"""

import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Module-level singleton for settings
_settings_cache = None

# Required settings keys
REQUIRED_SETTINGS = ['spreadsheet_id', 'credentials_file']

def get_config_path(config_file=None):
    """
    Get the path to the configuration file.

    Args:
        config_file: Path to configuration file (default: ~/.depensage/config.json)

    Returns:
        Path to configuration file
    """
    if config_file:
        return config_file

    home_dir = str(Path.home())
    config_dir = os.path.join(home_dir, '.depensage')
    return os.path.join(config_dir, 'config.json')

def load_settings(config_file=None, force_reload=False):
    """
    Load settings from configuration file, using cached version if available.

    Args:
        config_file: Path to configuration file (default: ~/.depensage/config.json)
        force_reload: Whether to force reloading from file, ignoring cache

    Returns:
        Dictionary with configuration settings

    Raises:
        ValueError: If required settings are missing or credentials file doesn't exist
    """
    global _settings_cache

    # Return cached settings if available and not forcing reload
    if _settings_cache is not None and not force_reload:
        return _settings_cache

    # Get config file path
    config_path = get_config_path(config_file)

    # Start with empty settings
    settings = {}

    if not os.path.exists(config_path):
        raise ValueError(f"Config file {config_path} does not exist")
    try:
        with open(config_path, 'r') as f:
            settings = json.load(f)
        logger.info(f"Loaded settings from {config_path}")

        missing_keys = []
        for key in REQUIRED_SETTINGS:
            if not settings.get(key):
                missing_keys.append(key)

        if missing_keys:
            raise ValueError(f"Missing required settings: {', '.join(missing_keys)}")

        creds_file = settings.get('credentials_file')
        if not os.path.exists(creds_file):
            raise ValueError(f"Credentials file does not exist: {creds_file}")

    except ValueError:
        # Re-raise ValueError exceptions
        raise
    except Exception as e:
        logger.error(f"Failed to load settings from {config_path}: {e}")

    # Cache the settings
    _settings_cache = settings

    return settings
