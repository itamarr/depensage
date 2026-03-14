"""Unit tests for CC charge verification."""

import unittest
from datetime import datetime
from unittest.mock import MagicMock

from depensage.engine.bank_parser import CCLumpSum
from depensage.engine.verification import (
    verify_cc_charges,
    group_lump_sums_by_billing_month,
    _parse_expense_rows_with_status,
    format_verification_report,
)


class TestGroupLumpSums(unittest.TestCase):

    def test_groups_by_month(self):
        lumps = [
            CCLumpSum(date=datetime(2026, 3, 10), amount=3000),
            CCLumpSum(date=datetime(2026, 3, 10), amount=9000),
            CCLumpSum(date=datetime(2026, 2, 10), amount=11000),
        ]
        groups = group_lump_sums_by_billing_month(lumps)
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[(2026, 3)]), 2)
        self.assertEqual(len(groups[(2026, 2)]), 1)

    def test_empty_list(self):
        groups = group_lump_sums_by_billing_month([])
        self.assertEqual(len(groups), 0)


class TestParseExpenseRows(unittest.TestCase):

    def test_separates_by_status(self):
        rows = [
            ["Shop A", "", "food", 100, "סופר", 46100, "CC"],
            ["Shop B", "", "fuel", 200, "נסיעות", 46115, ""],
            ["Mortgage", "", "", 5000, "משכנתא", 46100, "BANK"],
        ]
        charged, pending, bank = _parse_expense_rows_with_status(rows)
        self.assertEqual(len(charged), 1)
        self.assertEqual(charged[0][1], 100)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0][1], 200)
        self.assertEqual(len(bank), 1)
        self.assertEqual(bank[0][1], 5000)

    def test_empty_status_is_pending(self):
        rows = [
            ["Shop", "", "", 50, "שונות", 46100, ""],
            ["Shop", "", "", 75, "שונות", 46100],  # no status column at all
        ]
        charged, pending, bank = _parse_expense_rows_with_status(rows)
        self.assertEqual(len(pending), 2)

    def test_skips_short_rows(self):
        rows = [["too", "short"]]
        charged, pending, bank = _parse_expense_rows_with_status(rows)
        self.assertEqual(len(charged) + len(pending) + len(bank), 0)

    def test_skips_zero_amounts(self):
        rows = [["Shop", "", "", 0, "cat", 46100, "CC"]]
        charged, pending, bank = _parse_expense_rows_with_status(rows)
        self.assertEqual(len(charged), 0)


class TestVerifyCCCharges(unittest.TestCase):

    def _make_handler(self, month_rows):
        """Create a mock handler that returns rows by month name."""
        handler = MagicMock()
        handler.read_expense_rows_with_status.side_effect = (
            lambda m: month_rows.get(m, [])
        )
        return handler

    def test_exact_match(self):
        """Lump sum matches charged + pending exactly."""
        handler = self._make_handler({
            "March": [
                # Charged in March (date <= 10)
                ["Shop", "", "", 500, "סופר", 46100, "CC"],
                ["Shop", "", "", 300, "שונות", 46100, "CC"],
            ],
            "February": [
                # Pending from February (date > 10)
                ["Shop", "", "", 200, "סופר", 46115, ""],
            ],
        })
        lumps = [CCLumpSum(date=datetime(2026, 3, 10), amount=1000)]
        result = verify_cc_charges(handler, lumps, 2026)

        self.assertEqual(len(result.cycles), 1)
        cycle = result.cycles[0]
        self.assertEqual(cycle.billing_month, "March")
        self.assertAlmostEqual(cycle.charged_total, 800)
        self.assertAlmostEqual(cycle.pending_total, 200)
        self.assertAlmostEqual(cycle.lump_total, 1000)
        self.assertTrue(cycle.matched)
        self.assertTrue(result.all_matched)

    def test_mismatch(self):
        """Lump sum doesn't match → mismatch flagged."""
        handler = self._make_handler({
            "February": [
                ["Shop", "", "", 500, "סופר", 46100, "CC"],
            ],
            "January": [],
        })
        lumps = [CCLumpSum(date=datetime(2026, 2, 10), amount=1000)]
        result = verify_cc_charges(handler, lumps, 2026)

        self.assertEqual(len(result.cycles), 1)
        self.assertFalse(result.cycles[0].matched)
        self.assertAlmostEqual(result.cycles[0].difference, 500)
        self.assertFalse(result.all_matched)

    def test_multiple_lump_sums_same_month(self):
        """Two cards' lump sums in same month are summed together."""
        handler = self._make_handler({
            "March": [
                ["Shop", "", "", 5000, "סופר", 46100, "CC"],
            ],
            "February": [
                ["Shop", "", "", 7000, "שונות", 46115, ""],
            ],
        })
        lumps = [
            CCLumpSum(date=datetime(2026, 3, 10), amount=3000),
            CCLumpSum(date=datetime(2026, 3, 10), amount=9000),
        ]
        result = verify_cc_charges(handler, lumps, 2026)

        cycle = result.cycles[0]
        self.assertAlmostEqual(cycle.lump_total, 12000)
        self.assertAlmostEqual(cycle.expected_total, 12000)
        self.assertTrue(cycle.matched)

    def test_bank_expenses_excluded(self):
        """BANK status expenses are not counted in verification."""
        handler = self._make_handler({
            "February": [
                ["CC Shop", "", "", 500, "סופר", 46100, "CC"],
                ["Bank charge", "", "", 3000, "משכנתא", 46100, "BANK"],
            ],
            "January": [
                ["Pending", "", "", 200, "שונות", 46115, ""],
                ["Bank", "", "", 1000, "חשבונות", 46115, "BANK"],
            ],
        })
        lumps = [CCLumpSum(date=datetime(2026, 2, 10), amount=700)]
        result = verify_cc_charges(handler, lumps, 2026)

        cycle = result.cycles[0]
        self.assertAlmostEqual(cycle.charged_total, 500)
        self.assertAlmostEqual(cycle.pending_total, 200)
        self.assertTrue(cycle.matched)

    def test_empty_lump_sums(self):
        """No lump sums → empty result."""
        handler = MagicMock()
        result = verify_cc_charges(handler, [], 2026)
        self.assertEqual(len(result.cycles), 0)
        self.assertTrue(result.all_matched)

    def test_january_with_prev_year_handler(self):
        """January billing cycle reads December from previous year handler."""
        handler_2026 = self._make_handler({
            "January": [
                ["Shop", "", "", 300, "סופר", 46100, "CC"],
            ],
        })
        handler_2025 = self._make_handler({
            "December": [
                ["Shop", "", "", 700, "שונות", 46115, ""],
            ],
        })
        lumps = [CCLumpSum(date=datetime(2026, 1, 11), amount=1000)]
        result = verify_cc_charges(
            handler_2026, lumps, 2026,
            prev_year_handler=handler_2025,
        )

        cycle = result.cycles[0]
        self.assertAlmostEqual(cycle.charged_total, 300)
        self.assertAlmostEqual(cycle.pending_total, 700)
        self.assertTrue(cycle.matched)


class TestFormatReport(unittest.TestCase):

    def test_format_matched(self):
        from depensage.engine.verification import VerificationResult, BillingCycleVerification
        result = VerificationResult(cycles=[
            BillingCycleVerification(
                billing_month="March", billing_year=2026,
                lump_sums=[], lump_total=1000,
                charged_total=800, pending_total=200,
                expected_total=1000, difference=0, matched=True,
            ),
        ])
        report = format_verification_report(result)
        self.assertIn("[OK]", report)
        self.assertNotIn("Difference", report)

    def test_format_mismatch(self):
        from depensage.engine.verification import VerificationResult, BillingCycleVerification
        result = VerificationResult(cycles=[
            BillingCycleVerification(
                billing_month="February", billing_year=2026,
                lump_sums=[], lump_total=1000,
                charged_total=500, pending_total=200,
                expected_total=700, difference=300, matched=False,
            ),
        ], all_matched=False)
        report = format_verification_report(result)
        self.assertIn("[MISMATCH]", report)
        self.assertIn("Difference", report)


if __name__ == "__main__":
    unittest.main()
