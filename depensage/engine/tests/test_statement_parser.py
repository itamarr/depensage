"""
Unit tests for the statement parser module.
"""

import unittest
import os
import tempfile
import pandas as pd
from datetime import datetime
from unittest.mock import patch

from depensage.engine.statement_parser import StatementParser


class TestStatementParser(unittest.TestCase):

    def setUp(self):
        self.parser = StatementParser()

        # Sample CSV matching real Israeli CC format (header row + data header + data)
        self.sample_csv = (
            "Header row to skip\n"
            "תאריך,עסק,סכום\n"
            "01/02/24,Supermarket,100.50\n"
            "05/02/24,Restaurant,50.75\n"
            "10/02/24,Gas Station,30.25\n"
        )

        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        self.temp_file.write(self.sample_csv.encode("utf-8-sig"))
        self.temp_file.close()

    def tearDown(self):
        os.unlink(self.temp_file.name)

    def test_parse_csv(self):
        result = self.parser.parse_statement(self.temp_file.name)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        self.assertListEqual(list(result.columns), ["date", "business_name", "amount"])
        self.assertEqual(result.iloc[0]["business_name"], "Supermarket")
        self.assertAlmostEqual(result.iloc[0]["amount"], 100.50)

    def test_parse_nonexistent_file(self):
        result = self.parser.parse_statement("nonexistent.csv")
        self.assertIsNone(result)

    def test_merge_statements(self):
        df1 = pd.DataFrame({
            "date": pd.to_datetime(["2024-02-01", "2024-02-05"]),
            "business_name": ["Supermarket", "Restaurant"],
            "amount": [100.50, 50.75],
        })
        df2 = pd.DataFrame({
            "date": pd.to_datetime(["2024-02-03", "2024-02-08"]),
            "business_name": ["Gas Station", "Pharmacy"],
            "amount": [30.25, 25.00],
        })

        result = self.parser.merge_statements(df1, df2)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 4)
        self.assertEqual(result.iloc[0]["business_name"], "Supermarket")
        self.assertEqual(result.iloc[-1]["business_name"], "Pharmacy")

        # Empty
        result = self.parser.merge_statements(pd.DataFrame(), pd.DataFrame())
        self.assertIsNone(result)

        # With None
        result = self.parser.merge_statements(df1, None)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

    def test_filter_new_transactions(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-02-01", "2024-02-05", "2024-02-10", "2024-02-15"]),
            "business_name": ["Supermarket", "Restaurant", "Gas Station", "Pharmacy"],
            "amount": [100.50, 50.75, 30.25, 25.00],
        })

        result = self.parser.filter_new_transactions(df, datetime(2024, 2, 8))
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

        result = self.parser.filter_new_transactions(pd.DataFrame(), datetime(2024, 2, 8))
        self.assertIsNone(result)

        result = self.parser.filter_new_transactions(None, datetime(2024, 2, 8))
        self.assertIsNone(result)

    def test_format_transactions_for_sheet(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-02-01", "2024-02-05"]),
            "business_name": ["Supermarket", "Restaurant"],
            "amount": [100.50, 50.75],
            "category": ["Groceries", "Dining"],
            "subcategory": ["Food", "Restaurant"],
        })

        result = self.parser.format_transactions_for_sheet(df)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 7)
        self.assertEqual(result[0][1], "Supermarket")   # Business name
        self.assertEqual(result[0][3], "Food")           # Subcategory
        self.assertEqual(result[0][4], "100.50")         # Amount
        self.assertEqual(result[0][5], "Groceries")      # Category
        self.assertEqual(result[0][6], "02/01/2024")     # Date

        self.assertEqual(self.parser.format_transactions_for_sheet(pd.DataFrame()), [])
        self.assertEqual(self.parser.format_transactions_for_sheet(None), [])


if __name__ == "__main__":
    unittest.main()
