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
            "spreadsheets": {
                "2026": {"id": "spreadsheet_id", "year": 2026, "default": true},
                "2026_dev": {"id": "dev_id", "year": 2026},
                "template": {"id": "template_id"}
            },
            "credentials_file": ".secrets/credentials.json",
            "default_savings_goal": "דירה"
        }

    Each spreadsheet entry is a dict with:
        - "id" (required): Google Spreadsheet ID
        - "year" (optional): Year this spreadsheet covers
        - "default" (optional): If true, preferred when multiple entries share a year

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
            raise ValueError(
                "'spreadsheets' must be a dict of spreadsheet entries"
            )

        # Validate each entry is a dict with an 'id' field
        for key, entry in settings['spreadsheets'].items():
            if not isinstance(entry, dict) or 'id' not in entry:
                raise ValueError(
                    f"Spreadsheet entry '{key}' must be a dict with an 'id' field. "
                    f"Example: {{\"{key}\": {{\"id\": \"spreadsheet_id\", \"year\": 2026}}}}"
                )

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


def get_spreadsheet_entry(key, settings=None):
    """Get a spreadsheet entry by its config key.

    Args:
        key: Config key (e.g., "2026", "2026_dev", "template").
        settings: Settings dict (loads from config if None).

    Returns:
        Dict with 'id', optional 'year', optional 'default'.

    Raises:
        ValueError: If key not found in config.
    """
    if settings is None:
        settings = load_settings()
    entries = settings['spreadsheets']
    if key not in entries:
        available = ', '.join(sorted(entries.keys()))
        raise ValueError(
            f"Spreadsheet '{key}' not found. Available: {available}"
        )
    return entries[key]


def get_entries_for_year(year, settings=None):
    """Get all (key, entry) pairs for a given year.

    Args:
        year: Year as int.
        settings: Settings dict (loads from config if None).

    Returns:
        List of (key, entry) tuples where entry["year"] == year.
    """
    if settings is None:
        settings = load_settings()
    return [
        (k, e) for k, e in settings['spreadsheets'].items()
        if e.get('year') == year
    ]


def get_all_years(settings=None):
    """Get all unique years configured (sorted).

    Args:
        settings: Settings dict (loads from config if None).

    Returns:
        Sorted list of unique year ints.
    """
    if settings is None:
        settings = load_settings()
    return sorted(set(
        e['year'] for e in settings['spreadsheets'].values()
        if 'year' in e
    ))
