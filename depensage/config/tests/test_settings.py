"""
Unit tests for the settings module with module-level singleton pattern.
"""

import unittest
import os
import tempfile
import json
import shutil
from unittest.mock import patch

import depensage.config.settings
from depensage.config.settings import load_settings, get_config_path, REQUIRED_SETTINGS

class TestSettings(unittest.TestCase):
    """Test cases for the settings module functions."""

    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'config.json')

        # Create a mock credentials file
        self.credentials_file = os.path.join(self.test_dir, 'credentials.json')
        with open(self.credentials_file, 'w') as f:
            f.write('{}')

        depensage.config.settings._settings_cache = None

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir)

        depensage.config.settings._settings_cache = None

    def test_get_config_path(self):
        """Test getting configuration path."""
        # Test with explicit config file
        self.assertEqual(get_config_path(self.config_file), self.config_file)

        # Test default path
        default_path = get_config_path()
        self.assertTrue(default_path.endswith('/.depensage/config.json'))

    def test_load_nonexistent_file(self):
        """Test loading from a nonexistent file."""
        # File shouldn't exist yet
        self.assertFalse(os.path.exists(self.config_file))

        # Load should raise ValueError
        with self.assertRaises(ValueError) as context:
            load_settings(self.config_file)

        self.assertIn("does not exist", str(context.exception))

    def test_save_and_load(self):
        """Test saving and loading settings."""
        # Create test settings
        test_settings = {
            'spreadsheet_id': 'test_id',
            'credentials_file': self.credentials_file
        }

        # Write settings to file manually
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(test_settings, f)

        # Load settings
        settings = load_settings(self.config_file)

        # Check settings
        self.assertEqual(settings['spreadsheet_id'], 'test_id')
        self.assertEqual(settings['credentials_file'], self.credentials_file)

    def test_singleton_caching(self):
        """Test that settings are cached and reused."""
        # Create test settings
        test_settings = {
            'spreadsheet_id': 'test_id',
            'log_level': 'DEBUG'
        }

        # Write settings to file
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(test_settings, f)

        # Load settings first time
        settings1 = load_settings(self.config_file)
        self.assertEqual(settings1['spreadsheet_id'], 'test_id')

        # Modify file to have different settings
        modified_settings = {
            'spreadsheet_id': 'new_id',
            'log_level': 'INFO'
        }
        with open(self.config_file, 'w') as f:
            json.dump(modified_settings, f)

        # Load settings again - should get cached version
        settings2 = load_settings(self.config_file)
        self.assertEqual(settings2['spreadsheet_id'], 'test_id')  # Still old value

        # Force reload - should get new values
        settings3 = load_settings(self.config_file, force_reload=True)
        self.assertEqual(settings3['spreadsheet_id'], 'new_id')  # New value

    def test_load_invalid_json(self):
        """Test loading invalid JSON."""
        # Write invalid JSON to the config file
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            f.write('invalid { json')

        # Load should fail gracefully and return empty dict
        with self.assertRaises(ValueError) as context:
            load_settings(self.config_file)

        self.assertIn("Expecting value: line 1 column 1 (char 0)", str(context.exception))


    def test_load_missing_required_settings(self):
        """Test loading settings with missing required keys."""
        # Create test settings with missing required keys
        test_settings = {
            'log_level': 'DEBUG'  # Missing spreadsheet_id and credentials_file
        }

        # Write settings to file
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(test_settings, f)

        # Load should raise ValueError
        with self.assertRaises(ValueError) as context:
            load_settings(self.config_file)

        self.assertIn("Missing required settings", str(context.exception))

    def test_load_nonexistent_credentials_file(self):
        """Test loading settings with a nonexistent credentials file."""
        # Create test settings with nonexistent credentials file
        test_settings = {
            'spreadsheet_id': 'test_id',
            'credentials_file': '/nonexistent/path.json'
        }

        # Write settings to file
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(test_settings, f)

        # Load should raise ValueError
        with self.assertRaises(ValueError) as context:
            load_settings(self.config_file)

        self.assertIn("Credentials file does not exist", str(context.exception))


if __name__ == '__main__':
    unittest.main()
