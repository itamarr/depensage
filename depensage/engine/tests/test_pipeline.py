"""
Unit tests for the core pipeline module.
"""

import unittest
from unittest.mock import MagicMock
import os
import tempfile
import pandas as pd

from depensage.engine.pipeline import run_pipeline
from depensage.engine.bank_parser import BANK_TRANSCRIPT_SIGNATURE
from depensage.classifier.cc_lookup import ClassificationResult


def _write_excel(rows, headers, title_row="Account holder info"):
    """Write an Excel file matching Israeli CC format (title row + header + data)."""
    df = pd.DataFrame(rows, columns=headers)
    f = tempfile.NamedTemporaryFile(delete=False, dir=tempfile.gettempdir(), suffix=".xlsx")
    f.close()
    with pd.ExcelWriter(f.name, engine="openpyxl") as writer:
        title_df = pd.DataFrame([[title_row]])
        title_df.to_excel(writer, index=False, header=False, startrow=0)
        df.to_excel(writer, index=False, startrow=1)
    return f.name


def _write_bank_excel(data_rows):
    """Write an Excel file matching Israeli bank transcript format."""
    headers = ["תאריך", "הפעולה", "פרטים", "אסמכתא", "חובה", "זכות",
               "יתרה בש''ח", "תאריך ערך", "לטובת", "עבור"]
    ncols = len(headers)
    all_rows = [
        [""] * ncols,
        [""] * ncols,
        [BANK_TRANSCRIPT_SIGNATURE] + [""] * (ncols - 1),
        ["12-702-12345"] + [""] * (ncols - 1),
        headers,
    ]
    all_rows.extend(data_rows)
    f = tempfile.NamedTemporaryFile(delete=False, dir=tempfile.gettempdir(), suffix=".xlsx")
    f.close()
    with pd.ExcelWriter(f.name, engine="openpyxl") as writer:
        df = pd.DataFrame(all_rows)
        df.to_excel(writer, index=False, header=False)
    return f.name


def _setup_handler_mock(handler, sheet_exists=True):
    """Configure a mock handler with defaults for VM loading."""
    handler.sheet_exists.return_value = sheet_exists
    handler.find_section_marker.return_value = 131
    handler.read_expense_rows.return_value = []
    handler.read_income_rows.return_value = []
    handler.find_first_empty_expense_row.return_value = 2
    handler.find_first_empty_income_row.return_value = 140
    handler.read_section_range.return_value = (None, [])
    handler.insert_rows.return_value = True
    handler.write_expense_rows.return_value = True
    handler.write_income_rows.return_value = True
    handler.batch_update_cells.return_value = True
    handler.create_sheet_from_template.return_value = True


class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.handler = MagicMock()
        self.classifier = MagicMock()
        self.temp_files = []
        _setup_handler_mock(self.handler)

    def tearDown(self):
        for f in self.temp_files:
            os.unlink(f)

    def _excel(self, rows, headers=None, title_row="Account holder info"):
        if headers is None:
            headers = ["תאריך", "שם בית עסק", "סכום"]
        path = _write_excel(rows, headers, title_row)
        self.temp_files.append(path)
        return path

    def _setup_classifier_all_classified(self):
        def classify_fn(df):
            classified = df.copy()
            classified["category"] = "סופר"
            classified["subcategory"] = ""
            return ClassificationResult(
                classified=classified,
                unclassified=pd.DataFrame(),
            )
        self.classifier.classify.side_effect = classify_fn

    def _setup_classifier_none_classified(self):
        def classify_fn(df):
            return ClassificationResult(
                classified=pd.DataFrame(),
                unclassified=df.copy(),
            )
        self.classifier.classify.side_effect = classify_fn

    def test_happy_path(self):
        path = self._excel([
            ["02/01/2024", "Shop A", 100.50],
            ["02/05/2024", "Shop B", 50.75],
        ])
        self._setup_classifier_all_classified()
        staged = run_pipeline([path], self.handler, self.classifier)

        self.assertEqual(staged.total_parsed, 2)
        self.assertEqual(staged.in_process_skipped, 0)
        self.assertEqual(staged.classified, 2)
        self.assertEqual(staged.unclassified, 0)

        # Check staged data
        stages = staged.sorted_stages()
        self.assertEqual(len(stages), 1)
        self.assertEqual(len(stages[0].new_expenses), 2)

        # Commit and check
        result = staged.commit(self.handler)
        self.assertEqual(len(result.months), 1)
        self.assertEqual(result.months[0].written, 2)
        self.assertEqual(result.months[0].duplicates, 0)
        self.handler.write_expense_rows.assert_called_once()

    def test_all_in_process_filtered(self):
        path = self._excel(
            rows=[
                ["02/01/2024", "Shop A", 100.50, "1234", None],
                ["02/05/2024", "Shop B", 50.75, "1234", None],
            ],
            headers=["תאריך", "שם בית עסק", "סכום", "כרטיס", "מועד חיוב"],
        )
        self._setup_classifier_all_classified()
        staged = run_pipeline([path], self.handler, self.classifier)
        self.assertEqual(staged.in_process_skipped, 2)
        self.assertEqual(staged.classified, 0)

    def test_all_duplicates(self):
        path = self._excel([
            ["02/01/2024", "Shop A", 100.50],
        ])
        self._setup_classifier_all_classified()
        self.handler.read_expense_rows.return_value = [
            ["Shop A", "", "", "100.50", "סופר", "02/01/2024"],
        ]
        staged = run_pipeline([path], self.handler, self.classifier)
        result = staged.commit(self.handler)
        self.assertEqual(result.months[0].written, 0)
        self.assertEqual(result.months[0].duplicates, 1)
        self.handler.write_expense_rows.assert_not_called()

    def test_unclassified_written_with_empty_category(self):
        path = self._excel([
            ["02/01/2024", "Unknown Shop", 42.00],
        ])
        self._setup_classifier_none_classified()
        staged = run_pipeline([path], self.handler, self.classifier)
        self.assertEqual(staged.unclassified, 1)
        self.assertIn("Unknown Shop", staged.unclassified_merchants)

        # Check staged rows have empty category
        stages = staged.sorted_stages()
        self.assertEqual(stages[0].new_expenses[0][4], "")  # empty category

        result = staged.commit(self.handler)
        self.assertEqual(result.months[0].written, 1)

        written_rows = self.handler.write_expense_rows.call_args[0][2]
        self.assertEqual(written_rows[0][4], "")  # empty category

    def test_no_marker_raises(self):
        path = self._excel([
            ["02/01/2024", "Shop A", 100.50],
        ])
        self._setup_classifier_all_classified()
        self.handler.find_section_marker.return_value = None
        # read_section_range returns (None, []) so budget_marker_row = 0
        with self.assertRaises(ValueError) as ctx:
            run_pipeline([path], self.handler, self.classifier)
        self.assertIn("budget", str(ctx.exception).lower())

    def test_row_insertion_when_full(self):
        path = self._excel([
            ["02/01/2024", "Shop A", 100.50],
            ["02/02/2024", "Shop B", 200.00],
            ["02/03/2024", "Shop C", 300.00],
        ])
        self._setup_classifier_all_classified()
        # marker at 8: sum at 5, blank at 6, budget-marker-like at 7, marker at 8
        # data rows 2-4 (last_data = marker-3 = 5). first_empty=5 means 1 slot,
        # need 3, so insert 2. Insert at marker-4 = 4.
        self.handler.find_section_marker.return_value = 8
        self.handler.find_first_empty_expense_row.return_value = 5
        staged = run_pipeline([path], self.handler, self.classifier)

        # Check staged insert count
        stages = staged.sorted_stages()
        self.assertEqual(stages[0].expense_insert_needed, 2)
        self.assertEqual(len(stages[0].new_expenses), 3)

        # Commit performs the insert within data area (marker-4)
        result = staged.commit(self.handler)
        self.handler.insert_rows.assert_called_once_with("February", 4, 2)
        self.assertEqual(result.months[0].written, 3)

    def test_empty_file(self):
        path = self._excel([], headers=["תאריך", "שם בית עסק", "סכום"])
        staged = run_pipeline([path], self.handler, self.classifier)
        self.assertEqual(staged.total_parsed, 0)

    def test_multi_month(self):
        path = self._excel([
            ["01/15/2024", "Shop A", 100.50],
            ["02/05/2024", "Shop B", 50.75],
        ])
        self._setup_classifier_all_classified()
        staged = run_pipeline([path], self.handler, self.classifier)
        result = staged.commit(self.handler)
        self.assertEqual(len(result.months), 2)
        month_names = {m.month for m in result.months}
        self.assertEqual(month_names, {"January", "February"})

    def test_multiple_files(self):
        path1 = self._excel([["02/01/2024", "Shop A", 100.50]])
        path2 = self._excel([["02/05/2024", "Shop B", 50.75]])
        self._setup_classifier_all_classified()
        staged = run_pipeline([path1, path2], self.handler, self.classifier)
        result = staged.commit(self.handler)
        self.assertEqual(result.total_parsed, 2)
        self.assertEqual(result.months[0].written, 2)

    def test_year_filter(self):
        """Only transactions from the specified year are processed."""
        path = self._excel([
            ["12/15/2025", "Shop A", 100.50],
            ["01/05/2026", "Shop B", 50.75],
        ])
        self._setup_classifier_all_classified()
        staged = run_pipeline([path], self.handler, self.classifier, year=2026)
        result = staged.commit(self.handler)
        self.assertEqual(result.total_parsed, 2)
        self.assertEqual(len(result.months), 1)
        self.assertEqual(result.months[0].month, "January")
        self.assertEqual(result.months[0].year, 2026)
        self.assertEqual(result.months[0].written, 1)

    def test_year_filter_no_matches(self):
        """Year filter with no matching transactions returns early."""
        path = self._excel([
            ["12/15/2025", "Shop A", 100.50],
        ])
        self._setup_classifier_all_classified()
        staged = run_pipeline([path], self.handler, self.classifier, year=2026)
        self.assertEqual(staged.total_parsed, 1)
        self.assertEqual(staged.classified, 0)
        result = staged.commit(self.handler)
        self.assertEqual(len(result.months), 0)

    def test_handler_dict(self):
        """Handlers dict routes years to correct handlers."""
        path = self._excel([
            ["12/15/2025", "Shop A", 100.50],
            ["01/05/2026", "Shop B", 50.75],
        ])
        self._setup_classifier_all_classified()

        handler_2025 = MagicMock()
        _setup_handler_mock(handler_2025)

        handler_2026 = MagicMock()
        _setup_handler_mock(handler_2026)

        handlers = {2025: handler_2025, 2026: handler_2026}
        staged = run_pipeline([path], handlers, self.classifier)
        result = staged.commit(handlers)

        handler_2025.write_expense_rows.assert_called_once()
        handler_2026.write_expense_rows.assert_called_once()
        self.assertEqual(len(result.months), 2)
        years = {m.year for m in result.months}
        self.assertEqual(years, {2025, 2026})

    def test_handler_dict_missing_year(self):
        """Missing year in handler dict raises ValueError."""
        path = self._excel([
            ["01/05/2026", "Shop A", 50.75],
        ])
        self._setup_classifier_all_classified()
        handlers = {2025: self.handler}  # no 2026
        with self.assertRaises(ValueError) as ctx:
            run_pipeline([path], handlers, self.classifier)
        self.assertIn("2026", str(ctx.exception))

    def test_new_month_gets_is_new_flag(self):
        """Stages for new months have is_new=True."""
        path = self._excel([
            ["02/01/2024", "Shop A", 100.50],
        ])
        self._setup_classifier_all_classified()
        self.handler.sheet_exists.return_value = False
        staged = run_pipeline([path], self.handler, self.classifier)
        stages = staged.sorted_stages()
        self.assertTrue(stages[0].is_new)

    def test_existing_month_not_is_new(self):
        """Stages for existing months have is_new=False."""
        path = self._excel([
            ["02/01/2024", "Shop A", 100.50],
        ])
        self._setup_classifier_all_classified()
        self.handler.sheet_exists.return_value = True
        staged = run_pipeline([path], self.handler, self.classifier)
        stages = staged.sorted_stages()
        self.assertFalse(stages[0].is_new)

    def test_commit_creates_sheet_for_new_month(self):
        """Commit creates the sheet from template for new months."""
        path = self._excel([
            ["02/01/2024", "Shop A", 100.50],
        ])
        self._setup_classifier_all_classified()
        self.handler.sheet_exists.return_value = False
        staged = run_pipeline([path], self.handler, self.classifier)
        staged.commit(self.handler)
        self.handler.create_sheet_from_template.assert_called_once_with(
            "February"
        )


    def test_existing_month_with_empty_accumulated_gets_carryover(self):
        """Existing sheets with no carryover data still get carryover applied."""
        path = self._excel([
            ["02/01/2024", "Shop A", 100.50],
        ])
        self._setup_classifier_all_classified()
        self.handler.sheet_exists.return_value = True

        # Budget section with a CARRY line that has accumulated=0
        budget_rows = [
            [500, 0, 2000, 0, "", "סופר", ""],
            [100, 0, 500, 0, "תת", "שונות", "CARRY"],
        ]
        # Previous month (January) has remaining=200 for the CARRY line
        prev_budget_rows = [
            [500, 0, 2000, 0, "", "סופר", ""],
            [200, 0, 500, 0, "תת", "שונות", "CARRY"],
        ]

        def section_range(sheet, section, columns, end_section=None):
            if section == "budget" and columns == "B:H":
                if sheet == "January":
                    return (131, prev_budget_rows)
                return (131, budget_rows)
            return (None, [])

        self.handler.read_section_range.side_effect = section_range

        staged = run_pipeline([path], self.handler, self.classifier)

        # Carryover should have generated updates for February
        stages = staged.sorted_stages()
        feb_stage = [s for s in stages if s.month == "February"][0]
        self.assertFalse(feb_stage.is_new)  # sheet exists
        self.assertTrue(len(feb_stage.carryover_updates) > 0)


class TestPipelineBankTransactions(unittest.TestCase):
    """Test pipeline handling of bank transcript files."""

    def setUp(self):
        self.handler = MagicMock()
        self.classifier = MagicMock()
        self.bank_classifier = MagicMock()
        self.income_classifier = MagicMock()
        self.temp_files = []

        _setup_handler_mock(self.handler)
        # Default: sheets don't exist (bank tests often involve new months)
        self.handler.sheet_exists.return_value = False

        # Savings marker for income section
        def section_marker(sheet, section):
            if section == "budget":
                return 131
            if section == "savings":
                return 160
            if section == "income":
                return 135
            if section == "reconciliation":
                return 180
            return None
        self.handler.find_section_marker.side_effect = section_marker

        # CC classifier (no CC files in these tests)
        self.classifier.classify.return_value = ClassificationResult(
            classified=pd.DataFrame(), unclassified=pd.DataFrame()
        )

    def tearDown(self):
        for f in self.temp_files:
            os.unlink(f)

    def _bank_excel(self, rows):
        path = _write_bank_excel(rows)
        self.temp_files.append(path)
        return path

    def _setup_bank_classifier_all(self):
        def classify_fn(df):
            classified = df.copy()
            classified["category"] = "חשבונות"
            classified["subcategory"] = ""
            return ClassificationResult(
                classified=classified, unclassified=pd.DataFrame()
            )
        self.bank_classifier.classify.side_effect = classify_fn

    def _setup_income_classifier_all(self):
        def classify_fn(df):
            classified = df.copy()
            classified["category"] = "משכורת"
            classified["comments"] = classified["action"]
            return ClassificationResult(
                classified=classified, unclassified=pd.DataFrame()
            )
        self.income_classifier.classify.side_effect = classify_fn

    def test_bank_expenses_written(self):
        """Bank expenses are written with BANK status."""
        path = self._bank_excel([
            ["01/01/2026", "קופת חולים", "", "123", 300, "", 5000, "", "", ""],
        ])
        self._setup_bank_classifier_all()
        self._setup_income_classifier_all()

        staged = run_pipeline(
            [path], self.handler, self.classifier, year=2026,
            bank_classifier=self.bank_classifier,
            income_classifier=self.income_classifier,
        )

        self.assertEqual(staged.total_parsed, 1)
        stages = staged.sorted_stages()
        self.assertEqual(len(stages[0].new_expenses), 1)
        # Check BANK status in staged rows
        self.assertEqual(stages[0].new_expenses[0][6], "BANK")

        result = staged.commit(self.handler)
        self.assertEqual(len(result.months), 1)
        self.assertEqual(result.months[0].written, 1)
        written_rows = self.handler.write_expense_rows.call_args[0][2]
        self.assertEqual(written_rows[0][6], "BANK")  # col H

    def test_bank_income_written(self):
        """Bank income is written to the income section."""
        path = self._bank_excel([
            ["01/01/2026", "חברה א", "emp123", "456", "", 25000,
             30000, "", "", ""],
        ])
        self._setup_bank_classifier_all()
        self._setup_income_classifier_all()

        staged = run_pipeline(
            [path], self.handler, self.classifier, year=2026,
            bank_classifier=self.bank_classifier,
            income_classifier=self.income_classifier,
        )

        self.assertEqual(staged.total_parsed, 1)
        result = staged.commit(self.handler)
        self.assertEqual(len(result.months), 1)
        self.assertEqual(result.months[0].income_written, 1)
        self.handler.write_income_rows.assert_called_once()

    def test_cc_lump_sums_extracted(self):
        """CC lump sum charges are extracted but not written."""
        path = self._bank_excel([
            ["01/10/2026", "כרטיסי אשראי ל-10", "", "789", 5000, "",
             -5000, "", "", ""],
            ["01/01/2026", "קופת חולים", "", "123", 300, "", 5000, "", "", ""],
        ])
        self._setup_bank_classifier_all()
        self._setup_income_classifier_all()

        staged = run_pipeline(
            [path], self.handler, self.classifier, year=2026,
            bank_classifier=self.bank_classifier,
            income_classifier=self.income_classifier,
        )

        lump_amounts = [ls.amount for ls in staged.cc_lump_sums]
        self.assertIn(5000.0, lump_amounts)
        # Only the expense should be staged, not the CC lump sum
        stages = staged.sorted_stages()
        self.assertEqual(len(stages[0].new_expenses), 1)

    def test_mixed_cc_and_bank(self):
        """Pipeline handles both CC and bank files together."""
        cc_path = _write_excel(
            [["01/05/2026", "Shop A", 100.50]],
            ["תאריך", "שם בית עסק", "סכום"],
        )
        self.temp_files.append(cc_path)
        bank_path = self._bank_excel([
            ["01/15/2026", "תשלום חובה", "", "123", 400, "", 5000,
             "", "", ""],
        ])

        # CC classifier
        def cc_classify(df):
            classified = df.copy()
            classified["category"] = "סופר"
            classified["subcategory"] = ""
            return ClassificationResult(
                classified=classified, unclassified=pd.DataFrame()
            )
        self.classifier.classify.side_effect = cc_classify

        self._setup_bank_classifier_all()
        self._setup_income_classifier_all()

        staged = run_pipeline(
            [cc_path, bank_path], self.handler, self.classifier, year=2026,
            bank_classifier=self.bank_classifier,
            income_classifier=self.income_classifier,
        )

        self.assertEqual(staged.total_parsed, 2)
        self.assertEqual(staged.classified, 2)
        result = staged.commit(self.handler)
        self.assertEqual(result.months[0].written, 2)

    def test_bank_income_dedup(self):
        """Duplicate income transactions are not written again."""
        path = self._bank_excel([
            ["01/01/2026", "חברה א", "emp123", "456", "", 25000,
             30000, "", "", ""],
        ])
        self._setup_bank_classifier_all()
        self._setup_income_classifier_all()

        self.handler.sheet_exists.return_value = True
        # Existing income row matches
        self.handler.read_income_rows.return_value = [
            ["חברה א", "25000.00", "משכורת", "01/01/2026"],
        ]

        staged = run_pipeline(
            [path], self.handler, self.classifier, year=2026,
            bank_classifier=self.bank_classifier,
            income_classifier=self.income_classifier,
        )

        result = staged.commit(self.handler)
        self.assertEqual(result.months[0].income_written, 0)
        self.assertEqual(result.months[0].income_duplicates, 1)
        self.handler.write_income_rows.assert_not_called()

    def test_no_bank_classifiers_writes_empty_categories(self):
        """Bank transactions without classifiers get empty categories."""
        path = self._bank_excel([
            ["01/01/2026", "קופת חולים", "", "123", 300, "", 5000, "", "", ""],
        ])

        staged = run_pipeline(
            [path], self.handler, self.classifier, year=2026,
        )

        self.assertEqual(staged.total_parsed, 1)
        self.assertEqual(staged.unclassified, 1)
        # Should still stage with empty category
        result = staged.commit(self.handler)
        self.assertEqual(result.months[0].written, 1)

    def test_bank_balance_staged(self):
        """Bank balance from transcript is staged for the correct month."""
        path = self._bank_excel([
            ["01/15/2026", "הוראת קבע", "", "111", 500, "", 20000, "", "", ""],
            ["01/28/2026", "תשלום", "", "222", 700, "", 19300, "", "", ""],
        ])
        self._setup_bank_classifier_all()
        self._setup_income_classifier_all()

        staged = run_pipeline(
            [path], self.handler, self.classifier, year=2026,
            bank_classifier=self.bank_classifier,
            income_classifier=self.income_classifier,
        )

        stages = staged.sorted_stages()
        self.assertEqual(len(stages), 1)
        self.assertAlmostEqual(stages[0].bank_balance, 19300.0)

    def test_bank_balance_committed_via_label_scan(self):
        """Bank balance commit uses find_reconciliation_label_row."""
        path = self._bank_excel([
            ["01/15/2026", "הוראת קבע", "", "111", 500, "", 20000, "", "", ""],
        ])
        self._setup_bank_classifier_all()
        self._setup_income_classifier_all()

        self.handler.find_reconciliation_label_row.return_value = 175
        self.handler.update_cell.return_value = True

        staged = run_pipeline(
            [path], self.handler, self.classifier, year=2026,
            bank_classifier=self.bank_classifier,
            income_classifier=self.income_classifier,
        )

        staged.commit(self.handler)
        self.handler.find_reconciliation_label_row.assert_called_once_with(
            "January", 'כסף בעו"ש'
        )
        self.handler.batch_update_cells.assert_called_with(
            "January", [("E175", 20000.0)]
        )

    def test_bank_balance_skipped_when_label_not_found(self):
        """Bank balance write is skipped if label not found in reconciliation."""
        path = self._bank_excel([
            ["01/15/2026", "הוראת קבע", "", "111", 500, "", 20000, "", "", ""],
        ])
        self._setup_bank_classifier_all()
        self._setup_income_classifier_all()

        self.handler.find_reconciliation_label_row.return_value = None
        self.handler.update_cell.return_value = True

        staged = run_pipeline(
            [path], self.handler, self.classifier, year=2026,
            bank_classifier=self.bank_classifier,
            income_classifier=self.income_classifier,
        )

        staged.commit(self.handler)
        self.handler.update_cell.assert_not_called()


if __name__ == "__main__":
    unittest.main()
