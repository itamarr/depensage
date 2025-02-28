"""
Unit tests for the sheets utility functions.
"""

import unittest
from datetime import datetime

from depensage.sheets.sheet_utils import HebrewMonthUtils, find_first_empty_row


class TestHebrewMonthUtils(unittest.TestCase):
    """Test cases for the HebrewMonthUtils class."""

    def test_get_hebrew_month_name(self):
        """Test getting Hebrew month names from numbers."""
        self.assertEqual(HebrewMonthUtils.get_hebrew_month_name(1), 'ינואר')
        self.assertEqual(HebrewMonthUtils.get_hebrew_month_name(12), 'דצמבר')
        self.assertEqual(HebrewMonthUtils.get_hebrew_month_name(13), '')  # Invalid month

    def test_get_month_number(self):
        """Test getting month numbers from Hebrew names."""
        self.assertEqual(HebrewMonthUtils.get_month_number('ינואר'), 1)
        self.assertEqual(HebrewMonthUtils.get_month_number('דצמבר'), 12)
        self.assertIsNone(HebrewMonthUtils.get_month_number('invalid'))  # Invalid name

    def test_get_sheet_name_for_date(self):
        """Test getting sheet name for a date."""
        jan_date = datetime(2024, 1, 15)
        dec_date = datetime(2024, 12, 25)

        self.assertEqual(HebrewMonthUtils.get_sheet_name_for_date(jan_date), 'ינואר')
        self.assertEqual(HebrewMonthUtils.get_sheet_name_for_date(dec_date), 'דצמבר')

    def test_parse_hebrew_date(self):
        """Test parsing dates in Hebrew formats."""
        # Test MM/DD/YYYY format
        self.assertEqual(
            HebrewMonthUtils.parse_hebrew_date('01/15/2024'),
            datetime(2024, 1, 15)
        )

        # Test DD/MM/YYYY format
        self.assertEqual(
            HebrewMonthUtils.parse_hebrew_date('15/01/2024'),
            datetime(2024, 1, 15)
        )

        # Test invalid format
        self.assertIsNone(HebrewMonthUtils.parse_hebrew_date('2024-01-15'))

    def test_format_date_for_sheet(self):
        """Test formatting dates for Google Sheet."""
        date = datetime(2024, 1, 15)
        self.assertEqual(HebrewMonthUtils.format_date_for_sheet(date), '01/15/2024')


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
