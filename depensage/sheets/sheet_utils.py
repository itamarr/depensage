"""
Utility functions for working with sheets and Hebrew date/time formats.
"""

from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class HebrewMonthUtils:
    """
    Utility class for handling Hebrew month names and conversions.
    """

    # Hebrew month names dictionary
    HEBREW_MONTHS = {
        1: 'ינואר', 2: 'פברואר', 3: 'מרץ', 4: 'אפריל', 5: 'מאי', 6: 'יוני',
        7: 'יולי', 8: 'אוגוסט', 9: 'ספטמבר', 10: 'אוקטובר', 11: 'נובמבר', 12: 'דצמבר'
    }

    # Reverse mapping for lookup by name
    MONTH_TO_NUMBER = {v: k for k, v in HEBREW_MONTHS.items()}

    @classmethod
    def get_hebrew_month_name(cls, month_num):
        """
        Get Hebrew month name from month number.

        Args:
            month_num: Month number (1-12)

        Returns:
            Hebrew month name or empty string if not found.
        """
        return cls.HEBREW_MONTHS.get(month_num, '')

    @classmethod
    def get_month_number(cls, hebrew_month_name):
        """
        Get month number from Hebrew month name.

        Args:
            hebrew_month_name: Hebrew month name

        Returns:
            Month number (1-12) or None if not found.
        """
        return cls.MONTH_TO_NUMBER.get(hebrew_month_name)

    @classmethod
    def get_sheet_name_for_date(cls, date):
        """
        Get sheet name for a given date.

        Args:
            date: datetime object

        Returns:
            Sheet name (Hebrew month name)
        """
        month_num = date.month
        return cls.get_hebrew_month_name(month_num)

    @classmethod
    def parse_hebrew_date(cls, date_str):
        """
        Parse Hebrew formatted date strings.

        Args:
            date_str: Date string in various formats

        Returns:
            datetime object or None if parsing fails
        """
        try:
            # Try to parse as MM/DD/YYYY
            return datetime.strptime(date_str, '%m/%d/%Y')
        except ValueError:
            try:
                # Try to parse as DD/MM/YYYY
                return datetime.strptime(date_str, '%d/%m/%Y')
            except ValueError:
                logger.error(f"Failed to parse date: {date_str}")
                return None

    @classmethod
    def format_date_for_sheet(cls, date):
        """
        Format date for Google Sheet (MM/DD/YYYY).

        Args:
            date: datetime object

        Returns:
            Formatted date string
        """
        return date.strftime('%m/%d/%Y')


def find_first_empty_row(values):
    """
    Find the first empty row in a range of values.

    Args:
        values: List of row values from sheet

    Returns:
        Index of first empty row (1-based for Google Sheets API)
    """
    if not values:
        return 1

    return len(values) + 1
