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
REQUIRED_SETTINGS = ['spreadsheets', 'credentials_file']

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', '.secrets', 'config.json'
)


def get_config_path(config_file=None):
    """
    Get the path to the configuration file.

    Args:
        config_file: Path to configuration file (default: .secrets/config.json)

    Returns:
        Path to configuration file
    """
    if config_file:
        return config_file

    return os.path.abspath(DEFAULT_CONFIG_PATH)

def load_settings(config_file=None, force_reload=False):
    """
    Load settings from configuration file, using cached version if available.

    Config format:
        {
            "spreadsheets": {"2025": "spreadsheet_id_1", "2026": "spreadsheet_id_2"},
            "credentials_file": ".secrets/credentials.json"
        }

    Args:
        config_file: Path to configuration file (default: .secrets/config.json)
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

        if not isinstance(settings['spreadsheets'], dict):
            raise ValueError("'spreadsheets' must be a dict mapping year to spreadsheet ID")

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


def get_spreadsheet_id(year, settings=None):
    """Get the spreadsheet ID for a given year.

    Args:
        year: Year as int or string (e.g., 2026 or "2026").
        settings: Settings dict (loads from config if None).

    Returns:
        Spreadsheet ID string.

    Raises:
        ValueError: If no spreadsheet configured for the given year.
    """
    if settings is None:
        settings = load_settings()
    year_str = str(year)
    spreadsheets = settings['spreadsheets']
    if year_str not in spreadsheets:
        available = ', '.join(sorted(spreadsheets.keys()))
        raise ValueError(
            f"No spreadsheet configured for year {year_str}. "
            f"Available: {available}"
        )
    return spreadsheets[year_str]
