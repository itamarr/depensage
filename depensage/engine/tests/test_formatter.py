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
        self.assertEqual(result[0][6], "CHARGED")  # day 1 <= 10

    def test_unclassified_rows(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-02-01"]),
            "business_name": ["Unknown Shop"],
            "amount": [42.00],
            "category": [""],
            "subcategory": [""],
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
        })
        result = format_for_sheet(df)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][2], "")  # subcategory
        self.assertEqual(result[0][4], "")  # category

    def test_charged_status_before_billing_day(self):
        """Transactions on or before billing day are CHARGED."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-02-10"]),
            "business_name": ["Shop"],
            "amount": [10.00],
        })
        result = format_for_sheet(df, billing_day=10)
        self.assertEqual(result[0][6], "CHARGED")

    def test_pending_status_after_billing_day(self):
        """Transactions after billing day are pending (empty status)."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-02-11"]),
            "business_name": ["Shop"],
            "amount": [10.00],
        })
        result = format_for_sheet(df, billing_day=10)
        self.assertEqual(result[0][6], "")

    def test_custom_billing_day(self):
        """Custom billing day is respected."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-02-15", "2024-02-16"]),
            "business_name": ["A", "B"],
            "amount": [10.00, 20.00],
        })
        result = format_for_sheet(df, billing_day=15)
        self.assertEqual(result[0][6], "CHARGED")  # day 15 <= 15
        self.assertEqual(result[1][6], "")  # day 16 > 15

    def test_mixed_charged_and_pending(self):
        """Mix of transactions before and after billing day."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-02-05", "2024-02-15", "2024-02-10"]),
            "business_name": ["A", "B", "C"],
            "amount": [10.00, 20.00, 30.00],
        })
        result = format_for_sheet(df)
        self.assertEqual(result[0][6], "CHARGED")  # day 5
        self.assertEqual(result[1][6], "")          # day 15
        self.assertEqual(result[2][6], "CHARGED")   # day 10 (exact)


if __name__ == "__main__":
    unittest.main()
