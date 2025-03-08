"""
Utility functions for working with sheets and date/time formats.
"""

from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SheetUtils:
    """
    Utility class for handling sheet operations and date conversions.
    """

    @classmethod
    def get_sheet_name_for_date(cls, date):
        """
        Get sheet name for a given date.

        Args:
            date: datetime object

        Returns:
            Sheet name (English month name)
        """
        month_name = date.strftime('%B')  # Get the full month name in English
        return month_name

    @classmethod
    def parse_date(cls, date_str):
        """
        Parse date strings in various formats.

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