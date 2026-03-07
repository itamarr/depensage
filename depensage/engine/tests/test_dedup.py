"""
Unit tests for the deduplication module.
"""

import unittest
import pandas as pd

from depensage.engine.dedup import deduplicate


class TestDeduplicate(unittest.TestCase):

    def _make_df(self, rows):
        """Helper: rows are (date_str, business_name, amount)."""
        return pd.DataFrame({
            "date": pd.to_datetime([r[0] for r in rows]),
            "business_name": [r[1] for r in rows],
            "amount": [r[2] for r in rows],
        })

    def _make_existing(self, rows):
        """Helper: rows are [business_name, notes, subcat, amount, category, date_str]."""
        return rows

    def test_all_new(self):
        new = self._make_df([
            ("2024-02-01", "Shop A", 100.50),
            ("2024-02-05", "Shop B", 50.75),
        ])
        existing = []
        result = deduplicate(new, existing)
        self.assertEqual(len(result), 2)

    def test_all_duplicates(self):
        new = self._make_df([
            ("2024-02-01", "Shop A", 100.50),
        ])
        existing = [
            ["Shop A", "", "", "100.50", "Food", "02/01/2024"],
        ]
        result = deduplicate(new, existing)
        self.assertEqual(len(result), 0)

    def test_mixed(self):
        new = self._make_df([
            ("2024-02-01", "Shop A", 100.50),
            ("2024-02-05", "Shop B", 50.75),
        ])
        existing = [
            ["Shop A", "", "", "100.50", "Food", "02/01/2024"],
        ]
        result = deduplicate(new, existing)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["business_name"], "Shop B")

    def test_same_name_date_different_amount(self):
        new = self._make_df([
            ("2024-02-01", "Shop A", 200.00),
        ])
        existing = [
            ["Shop A", "", "", "100.50", "Food", "02/01/2024"],
        ]
        result = deduplicate(new, existing)
        self.assertEqual(len(result), 1)

    def test_empty_new(self):
        result = deduplicate(pd.DataFrame(), [])
        self.assertTrue(result.empty)

    def test_none_new(self):
        result = deduplicate(None, [])
        self.assertIsNone(result)

    def test_existing_with_short_rows(self):
        """Existing rows with fewer than 6 columns should be skipped."""
        new = self._make_df([
            ("2024-02-01", "Shop A", 100.50),
        ])
        existing = [
            ["Shop A", "", ""],  # too short
        ]
        result = deduplicate(new, existing)
        self.assertEqual(len(result), 1)

    def test_amount_normalization(self):
        """Amounts with different string representations should match."""
        new = self._make_df([
            ("2024-02-01", "Shop A", 100.5),
        ])
        existing = [
            ["Shop A", "", "", "100.50", "Food", "02/01/2024"],
        ]
        result = deduplicate(new, existing)
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
