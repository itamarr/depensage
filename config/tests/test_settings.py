"""
Unit tests for the settings module.
"""

import unittest
import os
import tempfile
import json
import shutil
from unittest.mock import patch

from depensage.config.settings import Settings


class TestSettings(unittest.TestCase):
    """Test cases for the Settings class."""

    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'config.json')
        self.settings = Settings(self.config_file)

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir)

    def test_init_with_defaults(self):
        """Test initialization with default settings."""
        # Check that default settings were applied
        self.assertEqual(self.settings.get('model_dir'), 'models')
        self.assertEqual(self.settings.get('log_level'), 'INFO')
        self.assertEqual(self.settings.get('spreadsheet_id'), '')

    def test_load_nonexistent_file(self):
        """Test loading from a nonexistent file."""
        # File shouldn't exist yet
        self.assertFalse(os.path.exists(self.config_file))

        # Load should return False but not fail
        result = self.settings.load()
        self.assertFalse(result)

        # Settings should still have defaults
        self.assertEqual(self.settings.get('model_dir'), 'models')

    def test_save_and_load(self):
        """Test saving and loading settings."""
        # Change some settings
        self.settings.set('spreadsheet_id', 'test_id')
        self.settings.set('credentials_file', '/path/to/creds.json')

        # Check that file was created
        self.assertTrue(os.path.exists(self.config_file))

        # Create a new settings instance and load from the same file
        new_settings = Settings(self.config_file)

        # Check that settings were loaded correctly
        self.assertEqual(new_settings.get('spreadsheet_id'), 'test_id')
        self.assertEqual(new_settings.get('credentials_file'), '/path/to/creds.json')
        self.assertEqual(new_settings.get('model_dir'), 'models')  # Default should remain

    def test_set_multiple(self):
        """Test setting multiple settings at once."""
        # Set multiple settings
        new_settings = {
            'spreadsheet_id': 'new_id',
            'credentials_file': '/new/path.json',
            'log_level': 'DEBUG'
        }

        result = self.settings.set_multiple(new_settings)

        # Check result and values
        self.assertTrue(result)
        self.assertEqual(self.settings.get('spreadsheet_id'), 'new_id')
        self.assertEqual(self.settings.get('credentials_file'), '/new/path.json')
        self.assertEqual(self.settings.get('log_level'), 'DEBUG')
        self.assertEqual(self.settings.get('model_dir'), 'models')  # Unchanged default

    def test_reset(self):
        """Test resetting settings to defaults."""
        # Change some settings
        self.settings.set('spreadsheet_id', 'test_id')
        self.settings.set('log_level', 'DEBUG')

        # Reset settings
        result = self.settings.reset()

        # Check result and values
        self.assertTrue(result)
        self.assertEqual(self.settings.get('spreadsheet_id'), '')  # Back to default
        self.assertEqual(self.settings.get('log_level'), 'INFO')  # Back to default

    def test_get_with_default(self):
        """Test getting a setting with a custom default."""
        # Get a nonexistent setting with a default
        value = self.settings.get('nonexistent_key', 'default_value')

        # Check that default was returned
        self.assertEqual(value, 'default_value')

    def test_save_failure(self):
        """Test save failure handling."""
        # Mock open to raise an exception
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            result = self.settings.save()

            # Check that result indicates failure
            self.assertFalse(result)

    def test_load_invalid_json(self):
        """Test loading invalid JSON."""
        # Write invalid JSON to the config file
        with open(self.config_file, 'w') as f:
            f.write('invalid { json')

        # Create a new settings instance and try to load
        new_settings = Settings(self.config_file)

        # Load should fail but not raise exception
        result = new_settings.load()
        self.assertFalse(result)

        # Settings should still have defaults
        self.assertEqual(new_settings.get('model_dir'), 'models')


if __name__ == '__main__':
    unittest.main()
