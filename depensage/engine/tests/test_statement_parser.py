"""
Unit tests for the statement parser module.
"""

import unittest
import os
import tempfile
import pandas as pd

from depensage.engine.statement_parser import StatementParser


def _write_excel(rows, headers, title_row="Account holder info"):
    """Write an Excel file matching Israeli CC format (title row + header + data)."""
    df = pd.DataFrame(rows, columns=headers)
    # Prepend a title row by writing with startrow=1 and manually inserting title
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    f.close()
    with pd.ExcelWriter(f.name, engine="openpyxl") as writer:
        # Write title row at row 0
        title_df = pd.DataFrame([[title_row]])
        title_df.to_excel(writer, index=False, header=False, startrow=0)
        # Write header + data starting at row 1
        df.to_excel(writer, index=False, startrow=1)
    return f.name


class TestStatementParser(unittest.TestCase):

    def setUp(self):
        self.parser = StatementParser()
        self.temp_files = []

    def tearDown(self):
        for f in self.temp_files:
            os.unlink(f)

    def _excel(self, rows, headers, title_row="Account holder info"):
        path = _write_excel(rows, headers, title_row)
        self.temp_files.append(path)
        return path

    def test_parse_excel_basic(self):
        path = self._excel(
            rows=[
                ["01/02/2024", "Supermarket", 100.50],
                ["05/02/2024", "Restaurant", 50.75],
                ["10/02/2024", "Gas Station", 30.25],
            ],
            headers=["תאריך", "שם בית עסק", "סכום"],
        )
        result = self.parser.parse_statement(path)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        self.assertIn("date", result.columns)
        self.assertIn("business_name", result.columns)
        self.assertIn("amount", result.columns)
        self.assertEqual(result.iloc[0]["business_name"], "Supermarket")
        self.assertAlmostEqual(result.iloc[0]["amount"], 100.50)

    def test_no_charge_date_column(self):
        path = self._excel(
            rows=[["01/02/2024", "Shop", 10.00]],
            headers=["תאריך", "שם בית עסק", "סכום"],
        )
        result = self.parser.parse_statement(path)
        self.assertIsNotNone(result)
        self.assertNotIn("charge_date", result.columns)

    def test_charge_date_extracted(self):
        path = self._excel(
            rows=[
                ["01/02/2024", "Shop A", 100.50, "1234", "02/05/2024"],
                ["05/02/2024", "Shop B", 50.75, "1234", None],
            ],
            headers=["תאריך", "שם בית עסק", "סכום", "כרטיס", "מועד חיוב"],
        )
        result = self.parser.parse_statement(path)
        self.assertIsNotNone(result)
        self.assertIn("charge_date", result.columns)
        self.assertFalse(pd.isna(result.iloc[0]["charge_date"]))
        self.assertTrue(pd.isna(result.iloc[1]["charge_date"]))

    def test_rejects_csv(self):
        f = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        f.write(b"a,b,c\n1,2,3\n")
        f.close()
        self.temp_files.append(f.name)
        result = self.parser.parse_statement(f.name)
        self.assertIsNone(result)

    def test_parse_nonexistent_file(self):
        result = self.parser.parse_statement("nonexistent.xlsx")
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

        result = self.parser.merge_statements(pd.DataFrame(), pd.DataFrame())
        self.assertIsNone(result)

        result = self.parser.merge_statements(df1, None)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

    def test_filter_pending_removes_nat(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-02-01", "2024-02-05", "2024-02-10"]),
            "business_name": ["A", "B", "C"],
            "amount": [10, 20, 30],
            "charge_date": pd.to_datetime(["2024-03-01", pd.NaT, "2024-03-10"]),
        })
        result = StatementParser.filter_pending(df)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]["business_name"], "A")
        self.assertEqual(result.iloc[1]["business_name"], "C")

    def test_filter_pending_no_charge_date_column(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-02-01"]),
            "business_name": ["A"],
            "amount": [10],
        })
        result = StatementParser.filter_pending(df)
        self.assertEqual(len(result), 1)

    def test_filter_pending_empty(self):
        result = StatementParser.filter_pending(pd.DataFrame())
        self.assertTrue(result.empty)

    def test_filter_pending_none(self):
        result = StatementParser.filter_pending(None)
        self.assertIsNone(result)

    def test_filter_pending_all_pending(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-02-01", "2024-02-05"]),
            "business_name": ["A", "B"],
            "amount": [10, 20],
            "charge_date": [pd.NaT, pd.NaT],
        })
        result = StatementParser.filter_pending(df)
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
