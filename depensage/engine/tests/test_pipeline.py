"""
Unit tests for the core pipeline module.
"""

import unittest
from unittest.mock import MagicMock
import os
import tempfile
import pandas as pd

from depensage.engine.pipeline import run_pipeline
from depensage.classifier.lookup import ClassificationResult


def _write_excel(rows, headers, title_row="Account holder info"):
    """Write an Excel file matching Israeli CC format (title row + header + data)."""
    df = pd.DataFrame(rows, columns=headers)
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    f.close()
    with pd.ExcelWriter(f.name, engine="openpyxl") as writer:
        title_df = pd.DataFrame([[title_row]])
        title_df.to_excel(writer, index=False, header=False, startrow=0)
        df.to_excel(writer, index=False, startrow=1)
    return f.name


class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.handler = MagicMock()
        self.classifier = MagicMock()
        self.temp_files = []

        # Default handler behavior
        self.handler.get_or_create_month_sheet.return_value = "February"
        self.handler.find_section_marker.return_value = 131
        self.handler.read_expense_rows.return_value = []
        self.handler.find_first_empty_expense_row.return_value = 2
        self.handler.insert_rows.return_value = True
        self.handler.write_expense_rows.return_value = True

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
        result = run_pipeline([path], self.handler, self.classifier)

        self.assertEqual(result.total_parsed, 2)
        self.assertEqual(result.pending_skipped, 0)
        self.assertEqual(result.classified, 2)
        self.assertEqual(result.unclassified, 0)
        self.assertEqual(len(result.months), 1)
        self.assertEqual(result.months[0].written, 2)
        self.assertEqual(result.months[0].duplicates, 0)
        self.handler.write_expense_rows.assert_called_once()

    def test_all_pending_filtered(self):
        path = self._excel(
            rows=[
                ["02/01/2024", "Shop A", 100.50, "1234", None],
                ["02/05/2024", "Shop B", 50.75, "1234", None],
            ],
            headers=["תאריך", "שם בית עסק", "סכום", "כרטיס", "מועד חיוב"],
        )
        self._setup_classifier_all_classified()
        result = run_pipeline([path], self.handler, self.classifier)
        self.assertEqual(result.pending_skipped, 2)
        self.assertEqual(result.classified, 0)

    def test_all_duplicates(self):
        path = self._excel([
            ["02/01/2024", "Shop A", 100.50],
        ])
        self._setup_classifier_all_classified()
        self.handler.read_expense_rows.return_value = [
            ["Shop A", "", "", "100.50", "סופר", "02/01/2024"],
        ]
        result = run_pipeline([path], self.handler, self.classifier)
        self.assertEqual(result.months[0].written, 0)
        self.assertEqual(result.months[0].duplicates, 1)
        self.handler.write_expense_rows.assert_not_called()

    def test_unclassified_written_with_empty_category(self):
        path = self._excel([
            ["02/01/2024", "Unknown Shop", 42.00],
        ])
        self._setup_classifier_none_classified()
        result = run_pipeline([path], self.handler, self.classifier)
        self.assertEqual(result.unclassified, 1)
        self.assertEqual(result.months[0].written, 1)
        self.assertIn("Unknown Shop", result.unclassified_merchants)

        written_rows = self.handler.write_expense_rows.call_args[0][2]
        self.assertEqual(written_rows[0][4], "")  # empty category

    def test_no_marker_raises(self):
        path = self._excel([
            ["02/01/2024", "Shop A", 100.50],
        ])
        self._setup_classifier_all_classified()
        self.handler.find_section_marker.return_value = None
        with self.assertRaises(ValueError) as ctx:
            run_pipeline([path], self.handler, self.classifier)
        self.assertIn("marker", str(ctx.exception).lower())

    def test_row_insertion_when_full(self):
        path = self._excel([
            ["02/01/2024", "Shop A", 100.50],
            ["02/02/2024", "Shop B", 200.00],
            ["02/03/2024", "Shop C", 300.00],
        ])
        self._setup_classifier_all_classified()
        # marker at 6: total at 5, data rows 2-4 (3 slots)
        # first_empty=4 means 1 slot available, need 3, so insert 2
        self.handler.find_section_marker.return_value = 6
        self.handler.find_first_empty_expense_row.return_value = 4
        result = run_pipeline([path], self.handler, self.classifier)
        # insert before total row (marker-1=5)
        self.handler.insert_rows.assert_called_once_with("February", 5, 2)
        self.assertEqual(result.months[0].written, 3)

    def test_empty_file(self):
        path = self._excel([], headers=["תאריך", "שם בית עסק", "סכום"])
        result = run_pipeline([path], self.handler, self.classifier)
        self.assertEqual(result.total_parsed, 0)

    def test_multi_month(self):
        path = self._excel([
            ["01/15/2024", "Shop A", 100.50],
            ["02/05/2024", "Shop B", 50.75],
        ])
        self._setup_classifier_all_classified()
        self.handler.get_or_create_month_sheet.side_effect = lambda d: d.strftime("%B")
        result = run_pipeline([path], self.handler, self.classifier)
        self.assertEqual(len(result.months), 2)
        month_names = {m.month for m in result.months}
        self.assertEqual(month_names, {"January", "February"})

    def test_multiple_files(self):
        path1 = self._excel([["02/01/2024", "Shop A", 100.50]])
        path2 = self._excel([["02/05/2024", "Shop B", 50.75]])
        self._setup_classifier_all_classified()
        result = run_pipeline([path1, path2], self.handler, self.classifier)
        self.assertEqual(result.total_parsed, 2)
        self.assertEqual(result.months[0].written, 2)

    def test_year_filter(self):
        """Only transactions from the specified year are processed."""
        path = self._excel([
            ["12/15/2025", "Shop A", 100.50],
            ["01/05/2026", "Shop B", 50.75],
        ])
        self._setup_classifier_all_classified()
        self.handler.get_or_create_month_sheet.side_effect = lambda d: d.strftime("%B")
        result = run_pipeline([path], self.handler, self.classifier, year=2026)
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
        result = run_pipeline([path], self.handler, self.classifier, year=2026)
        self.assertEqual(result.total_parsed, 1)
        self.assertEqual(result.classified, 0)
        self.assertEqual(len(result.months), 0)

    def test_handler_dict(self):
        """Handlers dict routes years to correct handlers."""
        path = self._excel([
            ["12/15/2025", "Shop A", 100.50],
            ["01/05/2026", "Shop B", 50.75],
        ])
        self._setup_classifier_all_classified()

        handler_2025 = MagicMock()
        handler_2025.get_or_create_month_sheet.return_value = "December"
        handler_2025.find_section_marker.return_value = 131
        handler_2025.read_expense_rows.return_value = []
        handler_2025.find_first_empty_expense_row.return_value = 2
        handler_2025.write_expense_rows.return_value = True

        handler_2026 = MagicMock()
        handler_2026.get_or_create_month_sheet.return_value = "January"
        handler_2026.find_section_marker.return_value = 131
        handler_2026.read_expense_rows.return_value = []
        handler_2026.find_first_empty_expense_row.return_value = 2
        handler_2026.write_expense_rows.return_value = True

        handlers = {2025: handler_2025, 2026: handler_2026}
        result = run_pipeline([path], handlers, self.classifier)

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


if __name__ == "__main__":
    unittest.main()
