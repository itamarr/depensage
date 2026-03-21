"""Unit tests for the virtual_month module."""

import unittest
from unittest.mock import MagicMock

from depensage.engine.virtual_month import (
    BudgetLine, SavingsLine, VirtualMonth,
    load_from_sheet, load_from_template,
    compute_budget_remaining, compute_savings_total, compute_income_total,
)
from depensage.engine.carryover import (
    CarryoverResult, compute_carryover, apply_carryover_to_vm,
)


def _mock_handler(markers=None, budget_rows=None, savings_rows=None,
                  income_rows=None, expense_rows=None,
                  income_data_rows=None, sheet_exists=True):
    """Create a mock handler with section data."""
    handler = MagicMock()
    handler.sheet_exists.return_value = sheet_exists

    if markers is None:
        markers = {"budget": 131, "income": 135, "savings": 160,
                    "reconciliation": 180}

    handler.find_section_marker.side_effect = (
        lambda sheet, section: markers.get(section)
    )
    handler.read_expense_rows.return_value = expense_rows or []
    handler.read_income_rows.return_value = income_data_rows or []
    handler.find_first_empty_expense_row.return_value = 2
    handler.find_first_empty_income_row.return_value = 138

    def read_section_range(sheet_name, section, columns, end_section=None):
        if section == "budget" and columns == "B:H":
            return (markers.get("budget", 131), budget_rows or [])
        if section == "savings" and columns == "A:G":
            return (markers.get("savings", 160), savings_rows or [])
        if section == "income" and columns == "B:G":
            return (markers.get("income", 135), income_rows or [])
        return (None, [])

    handler.read_section_range.side_effect = read_section_range
    return handler


class TestLoadFromSheet(unittest.TestCase):

    def test_loads_budget_lines(self):
        budget_rows = [
            # B(remaining), C(expense), D(budget), E(accum), F(subcat), G(cat), H(carry)
            [500, 1500, 2000, 0, "", "סופר", ""],
            [300, 200, 500, 0, "", "חסכון", ""],
            [100, 400, 500, 0, "תת", "שונות", "CARRY"],
        ]
        handler = _mock_handler(budget_rows=budget_rows)
        vm = load_from_sheet(handler, "February", 2026)

        self.assertEqual(vm.month, "February")
        self.assertEqual(vm.year, 2026)
        self.assertFalse(vm.is_new)
        self.assertEqual(len(vm.budget_lines), 3)

        # Check savings budget detection
        self.assertEqual(vm.savings_budget_value, 500)

        # Check CARRY flag
        carry_lines = [l for l in vm.budget_lines if l.carry_flag]
        self.assertEqual(len(carry_lines), 1)
        self.assertEqual(carry_lines[0].category, "שונות")

    def test_loads_savings_lines(self):
        savings_rows = [
            # A(months), B(target), C(total), D(out), E(in), F(accum), G(goal)
            [12, 10000, 5000, 0, 2000, 3000, "רכב"],
            [0, 0, 0, 0, 0, 0, "קטגוריה"],  # header, skipped
            [24, 50000, 20000, 0, 0, 20000, "דירה"],
        ]
        handler = _mock_handler(savings_rows=savings_rows)
        vm = load_from_sheet(handler, "January", 2026)

        self.assertEqual(len(vm.savings_lines), 2)
        self.assertEqual(vm.savings_lines[0].goal_name, "רכב")
        self.assertEqual(vm.savings_lines[0].incoming, 2000)
        self.assertEqual(vm.savings_lines[1].goal_name, "דירה")

    def test_loads_income_total(self):
        income_rows = [
            ["", "", "", ""],
            ["הכנסות", "", "", ""],
            ["הערות", "כמה", "קטגוריה", "תאריך"],
            ["מעסיק", 25000, "משכורת", "01/01/2026"],
            ['סה"כ', "", "", 25000, "", ""],
        ]
        handler = _mock_handler(income_rows=income_rows)
        vm = load_from_sheet(handler, "January", 2026)
        self.assertEqual(vm.income_total, 25000)


class TestLoadFromTemplate(unittest.TestCase):

    def test_new_month_is_empty(self):
        handler = _mock_handler()
        vm = load_from_template(handler, "March", 2026)

        self.assertTrue(vm.is_new)
        self.assertEqual(vm.expense_rows, [])
        self.assertEqual(vm.income_rows, [])
        self.assertEqual(vm.first_empty_expense_row, 2)

    def test_template_income_start(self):
        handler = _mock_handler(
            markers={"budget": 131, "income": 135, "savings": 160,
                     "reconciliation": 180}
        )
        vm = load_from_template(handler, "March", 2026)
        self.assertEqual(vm.first_empty_income_row, 138)  # income + 3


class TestComputeFormulas(unittest.TestCase):

    def test_budget_remaining(self):
        line = BudgetLine(
            category="סופר", subcategory="", budget_amount=2000,
            accumulated=300, carry_flag=False, row_number=132,
        )
        self.assertEqual(compute_budget_remaining(line), 2300)

    def test_savings_total(self):
        line = SavingsLine(
            goal_name="רכב", target=10000, accumulated=5000,
            incoming=2000, outgoing=500, row_number=161,
        )
        self.assertEqual(compute_savings_total(line), 6500)

    def test_income_total(self):
        vm = VirtualMonth(month="Jan", year=2026, is_new=False)
        vm.income_rows = [
            ["מעסיק", 25000, "משכורת", "01/01"],
            ["בונוס", 5000, "מענק", "01/15"],
        ]
        self.assertEqual(compute_income_total(vm), 30000)

    def test_income_total_empty(self):
        vm = VirtualMonth(month="Jan", year=2026, is_new=True)
        self.assertEqual(compute_income_total(vm), 0.0)


class TestComputeCarryover(unittest.TestCase):

    def _make_source_vm(self, is_new=False):
        vm = VirtualMonth(month="January", year=2026, is_new=is_new)
        vm.budget_lines = [
            BudgetLine("סופר", "", 2000, 0, False, 132, remaining=500),
            BudgetLine("שונות", "תת", 500, 100, True, 133, remaining=200),
            BudgetLine("חסכון", "", 1000, 0, False, 134, remaining=1000),
        ]
        vm.savings_lines = [
            SavingsLine("רכב", 10000, 5000, 2000, 0, 161),
            SavingsLine("דירה", 50000, 20000, 0, 0, 162),
        ]
        vm.income_total = 30000
        vm.savings_budget_value = 1000
        vm.savings_budget_row = 134
        return vm

    def _make_dest_vm(self):
        vm = VirtualMonth(month="February", year=2026, is_new=True)
        vm.budget_lines = [
            BudgetLine("סופר", "", 2000, 0, False, 132),
            BudgetLine("שונות", "תת", 500, 0, True, 133),
            BudgetLine("חסכון", "", 0, 0, False, 134),
        ]
        vm.savings_lines = [
            SavingsLine("רכב", 10000, 0, 2000, 0, 161),
            SavingsLine("דירה", 50000, 0, 0, 0, 162),
        ]
        vm.savings_budget_row = 134
        return vm

    def test_same_spreadsheet_formulas(self):
        source = self._make_source_vm()
        dest = self._make_dest_vm()
        result = compute_carryover(source, dest, same_spreadsheet=True)

        # Budget: only CARRY line at row 133
        self.assertEqual(len(result.budget_updates), 1)
        ref, formula = result.budget_updates[0]
        self.assertEqual(ref, "E133")
        self.assertIn("January", formula)
        self.assertIn("MAX", formula)

        # In-memory value
        self.assertAlmostEqual(result.budget_accumulated[133], 200)

        # Savings: formulas for both goals
        self.assertEqual(len(result.savings_updates), 2)

        # Savings budget
        self.assertIsNotNone(result.savings_budget_update)
        self.assertEqual(result.savings_budget_update[0], "D134")

    def test_cross_year_static_values(self):
        source = self._make_source_vm()
        dest = self._make_dest_vm()
        result = compute_carryover(source, dest, same_spreadsheet=False)

        # Budget: static value for CARRY line
        self.assertEqual(len(result.budget_updates), 1)
        ref, val = result.budget_updates[0]
        self.assertEqual(ref, "E133")
        self.assertAlmostEqual(val, 200)  # remaining from source

    def test_apply_to_vm(self):
        source = self._make_source_vm()
        dest = self._make_dest_vm()
        result = compute_carryover(source, dest, same_spreadsheet=True)
        apply_carryover_to_vm(dest, result)

        # Budget accumulated set
        carry_line = [l for l in dest.budget_lines if l.carry_flag][0]
        self.assertAlmostEqual(carry_line.accumulated, 200)

        # Savings accumulated set
        car_goal = [l for l in dest.savings_lines
                    if l.goal_name == "רכב"][0]
        self.assertAlmostEqual(car_goal.accumulated, 7000)  # 5000 + 2000 - 0

        # Savings budget updated
        self.assertIsNotNone(dest.savings_budget_value)

    def test_no_income_skips_savings_budget(self):
        source = self._make_source_vm()
        source.income_total = None
        dest = self._make_dest_vm()
        result = compute_carryover(source, dest, same_spreadsheet=True)

        self.assertIsNone(result.savings_budget_update)
        self.assertIsNone(result.savings_budget)

    def test_uses_remaining_field_directly(self):
        """Carryover always uses line.remaining (reflects staged updates)."""
        source = self._make_source_vm(is_new=True)
        # remaining=200 was set on the CARRY line in _make_source_vm
        dest = self._make_dest_vm()
        result = compute_carryover(source, dest, same_spreadsheet=True)

        # Uses remaining=200 directly (not budget+accumulated)
        self.assertAlmostEqual(result.budget_accumulated[133], 200)


if __name__ == "__main__":
    unittest.main()
