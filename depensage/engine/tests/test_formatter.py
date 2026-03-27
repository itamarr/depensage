"""
Unit tests for the row formatter module.
"""

import unittest
import pandas as pd

from depensage.engine.formatter import format_for_sheet


class TestFormatForSheet(unittest.TestCase):

    def test_classified_rows(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-02-01", "2024-02-05"]),
            "business_name": ["Supermarket", "Restaurant"],
            "amount": [100.50, 50.75],
            "category": ["סופר", "בילויים וביזבוזים"],
            "subcategory": ["", "מסעדות"],
            "charge_date": pd.to_datetime(["2024-02-10", "2024-02-10"]),
        })
        result = format_for_sheet(df)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 7)
        # [business_name, notes, subcategory, amount, category, date, status]
        self.assertEqual(result[0][0], "Supermarket")
        self.assertEqual(result[0][1], "")  # notes
        self.assertEqual(result[0][2], "")  # subcategory
        self.assertEqual(result[0][3], "100.50")
        self.assertEqual(result[0][4], "סופר")
        self.assertEqual(result[0][5], "02/01/2024")
        self.assertEqual(result[0][6], "CC")  # charge_date in same month

    def test_unclassified_rows(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-02-01"]),
            "business_name": ["Unknown Shop"],
            "amount": [42.00],
            "category": [""],
            "subcategory": [""],
            "charge_date": pd.to_datetime(["2024-02-10"]),
        })
        result = format_for_sheet(df)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][4], "")  # empty category
        self.assertEqual(result[0][2], "")  # empty subcategory

    def test_empty_dataframe(self):
        self.assertEqual(format_for_sheet(pd.DataFrame()), [])

    def test_none_input(self):
        self.assertEqual(format_for_sheet(None), [])

    def test_no_category_columns(self):
        """DataFrame without category/subcategory columns should default to empty."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-02-01"]),
            "business_name": ["Shop"],
            "amount": [10.00],
            "charge_date": pd.to_datetime(["2024-02-10"]),
        })
        result = format_for_sheet(df)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][2], "")  # subcategory
        self.assertEqual(result[0][4], "")  # category

    def test_charged_when_charge_date_same_month(self):
        """Transactions charged in the same month get CC status."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-02-08"]),
            "business_name": ["Shop"],
            "amount": [10.00],
            "charge_date": pd.to_datetime(["2024-02-10"]),
        })
        result = format_for_sheet(df)
        self.assertEqual(result[0][6], "CC")

    def test_pending_when_charge_date_next_month(self):
        """Transactions charged next month get empty (pending) status."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-02-09"]),
            "business_name": ["Shop"],
            "amount": [10.00],
            "charge_date": pd.to_datetime(["2024-03-10"]),
        })
        result = format_for_sheet(df)
        self.assertEqual(result[0][6], "")

    def test_no_charge_date_column(self):
        """Without charge_date column, status defaults to empty."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-02-01"]),
            "business_name": ["Shop"],
            "amount": [10.00],
        })
        result = format_for_sheet(df)
        self.assertEqual(result[0][6], "")

    def test_mixed_charged_and_pending(self):
        """Mix of same-month and next-month charge dates."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-02-05", "2024-02-15", "2024-02-10"]),
            "business_name": ["A", "B", "C"],
            "amount": [10.00, 20.00, 30.00],
            "charge_date": pd.to_datetime(["2024-02-10", "2024-03-10", "2024-03-10"]),
        })
        result = format_for_sheet(df)
        self.assertEqual(result[0][6], "CC")  # charge_date Feb = tx Feb
        self.assertEqual(result[1][6], "")    # charge_date Mar ≠ tx Feb
        self.assertEqual(result[2][6], "")    # charge_date Mar ≠ tx Feb

    def test_cross_year_charge_date(self):
        """December transaction charged in January next year is pending."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-12-15"]),
            "business_name": ["Shop"],
            "amount": [10.00],
            "charge_date": pd.to_datetime(["2025-01-10"]),
        })
        result = format_for_sheet(df)
        self.assertEqual(result[0][6], "")  # different year+month


if __name__ == "__main__":
    unittest.main()
