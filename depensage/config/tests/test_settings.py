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
    load_settings, get_config_path, get_spreadsheet_entry,
    get_entries_for_year, get_all_years, REQUIRED_SETTINGS,
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
            'spreadsheets': {
                '2025': {'id': 'test_id_2025', 'year': 2025},
                '2026': {'id': 'test_id_2026', 'year': 2026},
            },
            'credentials_file': self.credentials_file,
        })

        settings = load_settings(self.config_file)
        self.assertEqual(settings['spreadsheets']['2025']['id'], 'test_id_2025')
        self.assertEqual(settings['spreadsheets']['2026']['id'], 'test_id_2026')
        self.assertEqual(settings['credentials_file'], self.credentials_file)

    def test_singleton_caching(self):
        """Test that settings are cached and reused."""
        self._write_config({
            'spreadsheets': {'2025': {'id': 'test_id'}},
            'credentials_file': self.credentials_file,
        })

        settings1 = load_settings(self.config_file)
        self.assertEqual(settings1['spreadsheets']['2025']['id'], 'test_id')

        # Modify file
        self._write_config({
            'spreadsheets': {'2025': {'id': 'new_id'}},
            'credentials_file': self.credentials_file,
        })

        # Should get cached version
        settings2 = load_settings(self.config_file)
        self.assertEqual(settings2['spreadsheets']['2025']['id'], 'test_id')

        # Force reload
        settings3 = load_settings(self.config_file, force_reload=True)
        self.assertEqual(settings3['spreadsheets']['2025']['id'], 'new_id')

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
            'spreadsheets': {'2025': {'id': 'test_id'}},
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

    def test_entry_must_have_id(self):
        """Test that each entry must have an 'id' field."""
        self._write_config({
            'spreadsheets': {'2025': {'year': 2025}},
            'credentials_file': self.credentials_file,
        })

        with self.assertRaises(ValueError) as context:
            load_settings(self.config_file)

        self.assertIn("must be a dict with an 'id' field", str(context.exception))

    def test_entry_must_be_dict(self):
        """Test that string entries (old format) are rejected."""
        self._write_config({
            'spreadsheets': {'2025': 'just_an_id'},
            'credentials_file': self.credentials_file,
        })

        with self.assertRaises(ValueError) as context:
            load_settings(self.config_file)

        self.assertIn("must be a dict with an 'id' field", str(context.exception))

    def test_get_spreadsheet_entry(self):
        """Test looking up spreadsheet entries by key."""
        settings = {
            'spreadsheets': {
                '2025': {'id': 'id_2025', 'year': 2025},
                '2026_dev': {'id': 'id_dev', 'year': 2026},
            },
        }

        entry = get_spreadsheet_entry('2025', settings)
        self.assertEqual(entry['id'], 'id_2025')
        self.assertEqual(entry['year'], 2025)

        entry = get_spreadsheet_entry('2026_dev', settings)
        self.assertEqual(entry['id'], 'id_dev')

        with self.assertRaises(ValueError) as context:
            get_spreadsheet_entry('nonexistent', settings)
        self.assertIn("not found", str(context.exception))
        self.assertIn("2025", str(context.exception))

    def test_get_entries_for_year(self):
        """Test finding entries by year."""
        settings = {
            'spreadsheets': {
                '2026_prod': {'id': 'id_prod', 'year': 2026, 'default': True},
                '2026_dev': {'id': 'id_dev', 'year': 2026},
                '2025': {'id': 'id_2025', 'year': 2025},
                'template': {'id': 'id_template'},
            },
        }

        entries_2026 = get_entries_for_year(2026, settings)
        self.assertEqual(len(entries_2026), 2)
        keys = {k for k, _ in entries_2026}
        self.assertEqual(keys, {'2026_prod', '2026_dev'})

        entries_2025 = get_entries_for_year(2025, settings)
        self.assertEqual(len(entries_2025), 1)

        entries_2024 = get_entries_for_year(2024, settings)
        self.assertEqual(len(entries_2024), 0)

    def test_get_all_years(self):
        """Test getting all configured years."""
        settings = {
            'spreadsheets': {
                '2026_prod': {'id': 'id_prod', 'year': 2026},
                '2026_dev': {'id': 'id_dev', 'year': 2026},
                '2025': {'id': 'id_2025', 'year': 2025},
                'template': {'id': 'id_template'},
            },
        }

        years = get_all_years(settings)
        self.assertEqual(years, [2025, 2026])

    def test_template_entry_no_year(self):
        """Template entries without year are valid and excluded from year queries."""
        self._write_config({
            'spreadsheets': {
                'template': {'id': 'id_template'},
                '2026': {'id': 'id_2026', 'year': 2026},
            },
            'credentials_file': self.credentials_file,
        })

        settings = load_settings(self.config_file)
        years = get_all_years(settings)
        self.assertEqual(years, [2026])

        entry = get_spreadsheet_entry('template', settings)
        self.assertEqual(entry['id'], 'id_template')
        self.assertNotIn('year', entry)


if __name__ == '__main__':
    unittest.main()
