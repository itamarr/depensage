"""
Settings manager for DepenSage.

This module handles loading, saving, and accessing configuration settings.
"""

import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Settings:
    """
    Configuration settings manager for DepenSage.
    """

    DEFAULT_SETTINGS = {
        'spreadsheet_id': '',
        'credentials_file': '',
        'model_dir': 'models',
        'log_level': 'INFO',
        'template_sheet_name': 'Month Template'
    }

    def __init__(self, config_file=None):
        """
        Initialize settings manager.

        Args:
            config_file: Path to configuration file (default: ~/.depensage/config.json)
        """
        if config_file is None:
            home_dir = str(Path.home())
            self.config_dir = os.path.join(home_dir, '.depensage')
            self.config_file = os.path.join(self.config_dir, 'config.json')
        else:
            self.config_file = config_file
            self.config_dir = os.path.dirname(os.path.abspath(config_file))

        # Ensure config directory exists
        os.makedirs(self.config_dir, exist_ok=True)

        # Initialize settings with defaults
        self.settings = self.DEFAULT_SETTINGS.copy()

        # Load settings from file if it exists
        self.load()

    def load(self):
        """
        Load settings from configuration file.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_settings = json.load(f)

                    # Update settings with loaded values
                    self.settings.update(loaded_settings)

                logger.info(f"Loaded settings from {self.config_file}")
                return True
            else:
                logger.info(f"Config file {self.config_file} does not exist, using defaults")
                return False
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            return False

    def save(self):
        """
        Save settings to configuration file.

        Returns:
            True if successful, False otherwise.
        """
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.settings, f, indent=4)

            logger.info(f"Saved settings to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            return False

    def get(self, key, default=None):
        """
        Get a setting value.

        Args:
            key: Setting key to retrieve.
            default: Default value if key doesn't exist.

        Returns:
            Setting value or default.
        """
        return self.settings.get(key, default)

    def set(self, key, value):
        """
        Set a setting value.

        Args:
            key: Setting key to set.
            value: Value to set.

        Returns:
            True if successful, False otherwise.
        """
        self.settings[key] = value
        return self.save()

    def set_multiple(self, settings_dict):
        """
        Set multiple settings at once.

        Args:
            settings_dict: Dictionary of settings to update.

        Returns:
            True if successful, False otherwise.
        """
        self.settings.update(settings_dict)
        return self.save()

    def reset(self):
        """
        Reset settings to defaults.

        Returns:
            True if successful, False otherwise.
        """
        self.settings = self.DEFAULT_SETTINGS.copy()
        return self.save()
