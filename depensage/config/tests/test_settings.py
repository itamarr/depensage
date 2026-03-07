"""
Unit tests for the settings module with module-level singleton pattern.
"""

import unittest
import os
import tempfile
import json
import shutil

import depensage.config.settings
from depensage.config.settings import (
    load_settings, get_config_path, get_spreadsheet_id, REQUIRED_SETTINGS,
)

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

    def _write_config(self, data):
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(data, f)

    def test_get_config_path(self):
        """Test getting configuration path."""
        # Test with explicit config file
        self.assertEqual(get_config_path(self.config_file), self.config_file)

        # Test default path
        default_path = get_config_path()
        self.assertTrue(default_path.endswith('.secrets/config.json'))

    def test_load_nonexistent_file(self):
        """Test loading from a nonexistent file."""
        self.assertFalse(os.path.exists(self.config_file))

        with self.assertRaises(ValueError) as context:
            load_settings(self.config_file)

        self.assertIn("does not exist", str(context.exception))

    def test_save_and_load(self):
        """Test saving and loading settings."""
        self._write_config({
            'spreadsheets': {'2025': 'test_id_2025', '2026': 'test_id_2026'},
            'credentials_file': self.credentials_file,
        })

        settings = load_settings(self.config_file)
        self.assertEqual(settings['spreadsheets']['2025'], 'test_id_2025')
        self.assertEqual(settings['spreadsheets']['2026'], 'test_id_2026')
        self.assertEqual(settings['credentials_file'], self.credentials_file)

    def test_singleton_caching(self):
        """Test that settings are cached and reused."""
        self._write_config({
            'spreadsheets': {'2025': 'test_id'},
            'credentials_file': self.credentials_file,
        })

        settings1 = load_settings(self.config_file)
        self.assertEqual(settings1['spreadsheets']['2025'], 'test_id')

        # Modify file
        self._write_config({
            'spreadsheets': {'2025': 'new_id'},
            'credentials_file': self.credentials_file,
        })

        # Should get cached version
        settings2 = load_settings(self.config_file)
        self.assertEqual(settings2['spreadsheets']['2025'], 'test_id')

        # Force reload
        settings3 = load_settings(self.config_file, force_reload=True)
        self.assertEqual(settings3['spreadsheets']['2025'], 'new_id')

    def test_load_invalid_json(self):
        """Test loading invalid JSON."""
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            f.write('invalid { json')

        with self.assertRaises(ValueError) as context:
            load_settings(self.config_file)

        self.assertIn("Expecting value: line 1 column 1 (char 0)", str(context.exception))

    def test_load_missing_required_settings(self):
        """Test loading settings with missing required keys."""
        self._write_config({'log_level': 'DEBUG'})

        with self.assertRaises(ValueError) as context:
            load_settings(self.config_file)

        self.assertIn("Missing required settings", str(context.exception))

    def test_load_nonexistent_credentials_file(self):
        """Test loading settings with a nonexistent credentials file."""
        self._write_config({
            'spreadsheets': {'2025': 'test_id'},
            'credentials_file': '/nonexistent/path.json',
        })

        with self.assertRaises(ValueError) as context:
            load_settings(self.config_file)

        self.assertIn("Credentials file does not exist", str(context.exception))

    def test_spreadsheets_must_be_dict(self):
        """Test that spreadsheets must be a dict, not a string."""
        self._write_config({
            'spreadsheets': 'just_a_string',
            'credentials_file': self.credentials_file,
        })

        with self.assertRaises(ValueError) as context:
            load_settings(self.config_file)

        self.assertIn("must be a dict", str(context.exception))

    def test_get_spreadsheet_id(self):
        """Test getting spreadsheet ID by year."""
        settings = {
            'spreadsheets': {'2025': 'id_2025', '2026': 'id_2026'},
            'credentials_file': self.credentials_file,
        }

        self.assertEqual(get_spreadsheet_id(2025, settings), 'id_2025')
        self.assertEqual(get_spreadsheet_id('2026', settings), 'id_2026')

        with self.assertRaises(ValueError) as context:
            get_spreadsheet_id(2024, settings)
        self.assertIn("No spreadsheet configured for year 2024", str(context.exception))
        self.assertIn("2025", str(context.exception))


if __name__ == '__main__':
    unittest.main()
