"""
Unit tests for the bank transcript parser module.
"""

import unittest
import os
import tempfile
import pandas as pd

from depensage.engine.bank_parser import (
    parse_bank_transcript,
    detect_bank_transcript,
    BANK_TRANSCRIPT_SIGNATURE,
    CC_CHARGE_ACTION,
)


def _write_bank_excel(rows, headers=None):
    """Write an Excel file matching Israeli bank transcript format.

    Rows 0-1: empty, Row 2: title, Row 3: account info, Row 4: headers, Row 5+: data.
    """
    if headers is None:
        headers = ["תאריך", "הפעולה", "פרטים", "אסמכתא", "חובה", "זכות",
                   "יתרה בש''ח", "תאריך ערך", "לטובת", "עבור"]

    f = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    f.close()

    ncols = len(headers)
    # Build raw rows: 0-1 empty, 2 title, 3 account, 4 headers, 5+ data
    all_rows = [
        [""] * ncols,
        [""] * ncols,
        [BANK_TRANSCRIPT_SIGNATURE] + [""] * (ncols - 1),
        ["12-702-12345"] + [""] * (ncols - 1),
        headers,
    ]
    all_rows.extend(rows)

    with pd.ExcelWriter(f.name, engine="openpyxl") as writer:
        df = pd.DataFrame(all_rows)
        df.to_excel(writer, index=False, header=False)

    return f.name


class TestBankParser(unittest.TestCase):

    def setUp(self):
        self.temp_files = []

    def tearDown(self):
        for f in self.temp_files:
            os.unlink(f)

    def _bank_excel(self, rows, headers=None):
        path = _write_bank_excel(rows, headers)
        self.temp_files.append(path)
        return path

    def test_parse_basic(self):
        """Parse a bank transcript with mixed debits, credits, and CC charges."""
        path = self._bank_excel([
            ["01/01/2026", "הלוואת דיור", "812529", "812529",
             5061.82, None, 50000, "01/01/2026", "", ""],
            ["01/01/2026", "חברה א", "123", "123",
             None, 27000.00, 77000, "01/01/2026", "", ""],
            ["01/10/2026", CC_CHARGE_ACTION, "", "8547994",
             9090.54, None, 67909.46, "01/10/2026", "", ""],
            ["01/10/2026", CC_CHARGE_ACTION, "", "8547994",
             4043.30, None, 63866.16, "01/10/2026", "", ""],
            ["01/05/2026", "קופת חולים", "health", "999",
             350.00, None, 49650, "01/05/2026", "", ""],
        ])

        result = parse_bank_transcript(path)
        self.assertIsNotNone(result)

        # Expenses: mortgage + maccabi (CC charges excluded)
        self.assertEqual(len(result.expenses), 2)
        self.assertAlmostEqual(result.expenses.iloc[0]["amount"], 5061.82)
        self.assertAlmostEqual(result.expenses.iloc[1]["amount"], 350.00)

        # Income: salary
        self.assertEqual(len(result.income), 1)
        self.assertAlmostEqual(result.income.iloc[0]["amount"], 27000.00)

        # CC lump sums: 2 charges with dates
        self.assertEqual(len(result.cc_lump_sums), 2)
        self.assertAlmostEqual(result.cc_lump_sums[0].amount, 9090.54)
        self.assertAlmostEqual(result.cc_lump_sums[1].amount, 4043.30)
        self.assertIsNotNone(result.cc_lump_sums[0].date)

    def test_expense_columns(self):
        """Expenses DataFrame has expected columns."""
        path = self._bank_excel([
            ["02/01/2026", "הלוואת דיור", "details", "ref123",
             5000.00, None, 50000, "02/01/2026", "", ""],
        ])
        result = parse_bank_transcript(path)
        expected_cols = {"date", "action", "details", "amount", "reference"}
        self.assertEqual(set(result.expenses.columns), expected_cols)

    def test_income_columns(self):
        """Income DataFrame has expected columns."""
        path = self._bank_excel([
            ["02/01/2026", "חברה א", "emp123", "ref",
             None, 25000.00, 75000, "02/01/2026", "", ""],
        ])
        result = parse_bank_transcript(path)
        expected_cols = {"date", "action", "details", "amount", "reference"}
        self.assertEqual(set(result.income.columns), expected_cols)

    def test_all_cc_charges(self):
        """Only CC charges → empty expenses and income, CC sums populated."""
        path = self._bank_excel([
            ["01/10/2026", CC_CHARGE_ACTION, "", "8547994",
             5000.00, None, 45000, "01/10/2026", "", ""],
        ])
        result = parse_bank_transcript(path)
        self.assertEqual(len(result.expenses), 0)
        self.assertEqual(len(result.income), 0)
        self.assertEqual(len(result.cc_lump_sums), 1)

    def test_no_cc_charges(self):
        """No CC charges → empty cc_lump_sums."""
        path = self._bank_excel([
            ["01/01/2026", "הלוואת דיור", "", "812529",
             5000.00, None, 50000, "01/01/2026", "", ""],
        ])
        result = parse_bank_transcript(path)
        self.assertEqual(len(result.cc_lump_sums), 0)
        self.assertEqual(len(result.expenses), 1)

    def test_empty_transcript(self):
        """Transcript with headers but no data rows."""
        path = self._bank_excel([])
        result = parse_bank_transcript(path)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.expenses), 0)
        self.assertEqual(len(result.income), 0)
        self.assertEqual(len(result.cc_lump_sums), 0)

    def test_detect_bank_transcript(self):
        """detect_bank_transcript identifies bank files correctly."""
        path = self._bank_excel([
            ["01/01/2026", "הלוואת דיור", "", "",
             5000.00, None, 50000, "01/01/2026", "", ""],
        ])
        self.assertTrue(detect_bank_transcript(path))

    def test_detect_rejects_cc_transcript(self):
        """detect_bank_transcript rejects CC statement files."""
        # CC format: title row + header at row 1
        f = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
        f.close()
        self.temp_files.append(f.name)
        df = pd.DataFrame([["01/01/2026", "Shop", 100.00]],
                          columns=["תאריך", "שם בית עסק", "סכום"])
        with pd.ExcelWriter(f.name, engine="openpyxl") as writer:
            title = pd.DataFrame([["Account holder info"]])
            title.to_excel(writer, index=False, header=False, startrow=0)
            df.to_excel(writer, index=False, startrow=1)
        self.assertFalse(detect_bank_transcript(f.name))

    def test_nonexistent_file(self):
        """Nonexistent file returns None."""
        result = parse_bank_transcript("nonexistent.xlsx")
        self.assertIsNone(result)

    def test_rejects_csv(self):
        """CSV file returns None."""
        f = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        f.write(b"a,b,c\n1,2,3\n")
        f.close()
        self.temp_files.append(f.name)
        self.assertIsNone(parse_bank_transcript(f.name))

    def test_dates_parsed_correctly(self):
        """Dates are parsed as datetime objects."""
        path = self._bank_excel([
            ["03/05/2026", "קופת חולים", "", "",
             200.00, None, 50000, "03/05/2026", "", ""],
        ])
        result = parse_bank_transcript(path)
        self.assertEqual(result.expenses.iloc[0]["date"].year, 2026)
        self.assertEqual(result.expenses.iloc[0]["date"].month, 3)

    def test_monthly_balances_extracted(self):
        """Monthly balances use the last balance per month."""
        path = self._bank_excel([
            ["01/01/2026", "הוראת קבע", "", "",
             2000, None, 40000, "01/01/2026", "", ""],
            ["01/15/2026", "העברה", "", "",
             500, None, 38000, "01/15/2026", "", ""],
            ["01/28/2026", "תשלום", "", "",
             700, None, 37300, "01/28/2026", "", ""],
            ["02/01/2026", "העברה נכנסת", "", "",
             None, 10000, 47300, "02/01/2026", "", ""],
        ])
        result = parse_bank_transcript(path)
        self.assertIn((2026, 1), result.monthly_balances)
        self.assertIn((2026, 2), result.monthly_balances)
        # January: last transaction is 01/28 with balance 37300
        self.assertAlmostEqual(result.monthly_balances[(2026, 1)], 37300)
        # February: only transaction is 02/01 with balance 47300
        self.assertAlmostEqual(result.monthly_balances[(2026, 2)], 47300)

    def test_monthly_balances_empty_when_no_balance_column(self):
        """No balance data returns empty dict."""
        # Use headers without balance column
        headers = ["תאריך", "הפעולה", "פרטים", "אסמכתא", "חובה", "זכות",
                   "תאריך ערך", "לטובת", "עבור"]
        path = self._bank_excel([
            ["01/01/2026", "תשלום", "", "", 500, None, "01/01/2026", "", ""],
        ], headers=headers)
        result = parse_bank_transcript(path)
        self.assertEqual(result.monthly_balances, {})


if __name__ == "__main__":
    unittest.main()
