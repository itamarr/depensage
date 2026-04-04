"""Unit tests for the staging module."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock

from depensage.engine.staging import StagedPipelineResult, MonthStage


class TestStagedPipelineResult(unittest.TestCase):

    def _make_staged(self):
        staged = StagedPipelineResult(
            total_parsed=5,
            in_process_skipped=1,
            classified=3,
            unclassified=1,
            unclassified_merchants=["Unknown Shop"],
        )
        stage = staged.get_or_create_stage("February", 2026)
        stage.new_expenses = [
            ["Shop A", "", "", "100.50", "סופר", "02/01/2026", "CC"],
            ["Shop B", "", "", "50.00", "שונות", "02/05/2026", "CC"],
        ]
        stage.expense_start_row = 2
        stage.duplicates = 1
        return staged

    def test_summary(self):
        staged = self._make_staged()
        summary = staged.summary()
        self.assertIn("Parsed:     5", summary)
        self.assertIn("2 expenses", summary)
        self.assertIn("1 duplicates", summary)
        self.assertIn("Unknown Shop", summary)

    def test_has_writes(self):
        staged = self._make_staged()
        self.assertTrue(staged.has_writes())

    def test_has_writes_empty(self):
        staged = StagedPipelineResult()
        self.assertFalse(staged.has_writes())

    def test_has_writes_only_duplicates(self):
        staged = StagedPipelineResult()
        stage = staged.get_or_create_stage("January", 2026)
        stage.duplicates = 5
        self.assertFalse(staged.has_writes())

    def test_export_xlsx(self):
        staged = self._make_staged()
        path = os.path.join(tempfile.mkdtemp(), "test_staged.xlsx")
        result_path = staged.export_xlsx(path)
        self.assertEqual(result_path, path)
        self.assertTrue(os.path.exists(path))
        os.unlink(path)

    def test_commit(self):
        staged = self._make_staged()
        handler = MagicMock()
        handler.write_expense_rows.return_value = True
        handler.find_section_marker.return_value = 131

        result = staged.commit(handler)
        self.assertEqual(len(result.months), 1)
        self.assertEqual(result.months[0].written, 2)
        self.assertEqual(result.months[0].duplicates, 1)
        handler.write_expense_rows.assert_called_once()

    def test_commit_with_insert(self):
        staged = StagedPipelineResult()
        stage = staged.get_or_create_stage("March", 2026)
        stage.new_expenses = [
            ["Shop", "", "", "200.00", "סופר", "03/01/2026", "CC"],
        ]
        stage.expense_start_row = 5
        stage.expense_insert_needed = 3

        handler = MagicMock()
        handler.write_expense_rows.return_value = True
        handler.find_section_marker.return_value = 20
        handler.insert_rows.return_value = True

        result = staged.commit(handler)
        handler.insert_rows.assert_called_once_with("March", 16, 3)  # marker-4 = 20-4
        handler.write_expense_rows.assert_called_once()
        self.assertEqual(result.months[0].written, 1)

    def test_commit_income(self):
        staged = StagedPipelineResult()
        stage = staged.get_or_create_stage("January", 2026)
        stage.new_income = [
            ["Salary", "25000.00", "משכורת", "01/01/2026"],
        ]
        stage.income_start_row = 140

        handler = MagicMock()
        handler.write_income_rows.return_value = True
        handler.find_section_marker.return_value = 160

        result = staged.commit(handler)
        handler.write_income_rows.assert_called_once()
        self.assertEqual(result.months[0].income_written, 1)

    def test_sorted_stages(self):
        staged = StagedPipelineResult()
        staged.get_or_create_stage("March", 2026)
        staged.get_or_create_stage("January", 2026)
        staged.get_or_create_stage("December", 2025)

        stages = staged.sorted_stages()
        self.assertEqual(
            [(s.month, s.year) for s in stages],
            [("December", 2025), ("January", 2026), ("March", 2026)],
        )


class TestStagedSummaryWithIncome(unittest.TestCase):

    def test_income_in_summary(self):
        staged = StagedPipelineResult(total_parsed=1)
        stage = staged.get_or_create_stage("January", 2026)
        stage.new_income = [["Salary", "25000.00", "משכורת", "01/01/2026"]]
        stage.income_duplicates = 1

        summary = staged.summary()
        self.assertIn("1 income", summary)
        self.assertIn("1 income dupes", summary)


if __name__ == "__main__":
    unittest.main()
