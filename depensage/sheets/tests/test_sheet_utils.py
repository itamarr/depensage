"""
Unit tests for the sheets utility functions.
"""

import unittest
from datetime import datetime

from depensage.sheets.sheet_utils import SheetUtils, find_first_empty_row

class TestSheetUtils(unittest.TestCase):
    """Test cases for the SheetUtils class."""

    def test_get_sheet_name_for_date(self):
        """Test getting sheet name for a date."""
        jan_date = datetime(2024, 1, 15)
        dec_date = datetime(2024, 12, 25)

        self.assertEqual(SheetUtils.get_sheet_name_for_date(jan_date), 'January')
        self.assertEqual(SheetUtils.get_sheet_name_for_date(dec_date), 'December')

    def test_parse_date(self):
        """Test parsing dates in various formats."""
        # Test MM/DD/YYYY format
        self.assertEqual(
            SheetUtils.parse_date('01/15/2024'),
            datetime(2024, 1, 15)
        )

        # Test DD/MM/YYYY format
        self.assertEqual(
            SheetUtils.parse_date('15/01/2024'),
            datetime(2024, 1, 15)
        )

        # Test invalid format
        self.assertIsNone(SheetUtils.parse_date('2024-01-15'))

    def test_format_date_for_sheet(self):
        """Test formatting dates for Google Sheet."""
        date = datetime(2024, 1, 15)
        self.assertEqual(SheetUtils.format_date_for_sheet(date), '01/15/2024')

class TestFindFirstEmptyRow(unittest.TestCase):
    """Test cases for the find_first_empty_row function."""

    def test_empty_values(self):
        """Test with empty values."""
        self.assertEqual(find_first_empty_row([]), 1)

    def test_non_empty_values(self):
        """Test with non-empty values."""
        values = [['a', 'b'], ['c', 'd'], ['e', 'f']]
        self.assertEqual(find_first_empty_row(values), 4)


if __name__ == '__main__':
    unittest.main()
