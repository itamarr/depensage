"""Unit tests for savings auto-allocation logic."""

import unittest

from depensage.engine.savings_allocator import (
    SavingsGoal, SavingsAllocation, SavingsAllocationResult,
    allocate_savings, BLATAM_GOAL_NAME,
)


def _goal(name, preset=0.0, outgoing=0.0, target=0.0, total=0.0, row=10):
    return SavingsGoal(
        goal_name=name,
        preset_incoming=preset,
        outgoing=outgoing,
        target=target,
        total=total,
        row_number=row,
    )


class TestAllocateSavings(unittest.TestCase):

    def test_good_month_surplus_to_default(self):
        """Budget > presets -> surplus goes to default goal."""
        goals = [
            _goal("רכב", preset=2000, row=50),
            _goal("חופשה", preset=1500, row=51),
            _goal("דירה", preset=0, row=52),
        ]
        result = allocate_savings(7000, goals, default_goal_name="דירה")

        self.assertIsNone(result.warning)
        self.assertFalse(result.zero_out)
        self.assertEqual(result.savings_budget, 7000)
        self.assertEqual(result.total_preset, 3500)
        self.assertEqual(result.surplus, 3500)

        alloc_map = {a.goal_name: a for a in result.allocations}
        self.assertEqual(alloc_map["רכב"].allocated, 2000)
        self.assertEqual(alloc_map["חופשה"].allocated, 1500)
        self.assertEqual(alloc_map["דירה"].allocated, 3500)  # 0 preset + 3500 surplus
        self.assertTrue(alloc_map["דירה"].is_default)

    def test_good_month_context_fields(self):
        """Context fields (preset, target, total) are preserved in allocations."""
        goals = [
            _goal("רכב", preset=2000, target=10000, total=5000, row=50),
        ]
        result = allocate_savings(5000, goals, default_goal_name="רכב")

        alloc = result.allocations[0]
        self.assertEqual(alloc.preset_incoming, 2000)
        self.assertEqual(alloc.target, 10000)
        self.assertEqual(alloc.current_total, 5000)

    def test_good_month_blatam_adjustment(self):
        """blatam with outgoing > 0 -> incoming increased to maintain target."""
        goals = [
            _goal("רכב", preset=2000, row=50),
            _goal(BLATAM_GOAL_NAME, preset=500, outgoing=300,
                  target=1000, total=800, row=51),
            _goal("דירה", preset=0, row=52),
        ]
        # Budget = 5000, presets = 2500, surplus = 2500
        # blatam needs: target(1000) - total(800) + preset(500) = 700
        # Extra for blatam: 700 - 500 = 200, remaining after: 2500 - 200 = 2300
        result = allocate_savings(5000, goals, default_goal_name="דירה")

        self.assertIsNone(result.warning)
        alloc_map = {a.goal_name: a for a in result.allocations}
        self.assertEqual(alloc_map[BLATAM_GOAL_NAME].allocated, 700)
        self.assertTrue(alloc_map[BLATAM_GOAL_NAME].is_blatam)
        self.assertEqual(alloc_map["דירה"].allocated, 2300)  # remaining surplus
        self.assertEqual(alloc_map["רכב"].allocated, 2000)

    def test_tight_month_warning(self):
        """0 < budget < presets -> warning, presets unchanged."""
        goals = [
            _goal("רכב", preset=2000, row=50),
            _goal("חופשה", preset=1500, row=51),
            _goal("דירה", preset=1000, row=52),
        ]
        result = allocate_savings(3000, goals, default_goal_name="דירה")

        self.assertIsNotNone(result.warning)
        self.assertIn("less than", result.warning)
        self.assertFalse(result.zero_out)
        self.assertEqual(result.total_preset, 4500)
        self.assertEqual(result.surplus, -1500)

        # Presets kept as-is
        alloc_map = {a.goal_name: a for a in result.allocations}
        self.assertEqual(alloc_map["רכב"].allocated, 2000)
        self.assertEqual(alloc_map["חופשה"].allocated, 1500)
        self.assertEqual(alloc_map["דירה"].allocated, 1000)

    def test_bad_month_zeroed(self):
        """Budget <= 0 -> all zeroed, warning set."""
        goals = [
            _goal("רכב", preset=2000, row=50),
            _goal("חופשה", preset=1500, row=51),
        ]
        result = allocate_savings(-500, goals, default_goal_name="רכב")

        self.assertIsNotNone(result.warning)
        self.assertIn("Negative", result.warning)
        self.assertTrue(result.zero_out)
        for a in result.allocations:
            self.assertEqual(a.allocated, 0.0)

    def test_bad_month_zero_budget(self):
        """Budget = 0 -> all zeroed."""
        goals = [_goal("רכב", preset=2000, row=50)]
        result = allocate_savings(0, goals)

        self.assertTrue(result.zero_out)
        self.assertEqual(result.allocations[0].allocated, 0.0)

    def test_no_blatam_goal(self):
        """No blatam row -> surplus goes straight to default."""
        goals = [
            _goal("רכב", preset=2000, row=50),
            _goal("דירה", preset=0, row=51),
        ]
        result = allocate_savings(5000, goals, default_goal_name="דירה")

        alloc_map = {a.goal_name: a for a in result.allocations}
        self.assertEqual(alloc_map["רכב"].allocated, 2000)
        self.assertEqual(alloc_map["דירה"].allocated, 3000)

    def test_default_goal_not_found_fallback(self):
        """Default goal name not in list -> falls back to first non-blatam goal."""
        goals = [
            _goal(BLATAM_GOAL_NAME, preset=500, row=50),
            _goal("רכב", preset=2000, row=51),
        ]
        result = allocate_savings(5000, goals, default_goal_name="nonexistent")

        alloc_map = {a.goal_name: a for a in result.allocations}
        # רכב becomes default fallback, gets surplus
        self.assertTrue(alloc_map["רכב"].is_default)
        self.assertEqual(alloc_map["רכב"].allocated, 4500)  # 2000 + 2500 surplus

    def test_no_default_goal_name(self):
        """default_goal_name=None -> falls back to first non-blatam goal."""
        goals = [
            _goal("רכב", preset=1000, row=50),
            _goal("חופשה", preset=500, row=51),
        ]
        result = allocate_savings(3000, goals, default_goal_name=None)

        alloc_map = {a.goal_name: a for a in result.allocations}
        self.assertTrue(alloc_map["רכב"].is_default)
        self.assertEqual(alloc_map["רכב"].allocated, 2500)  # 1000 + 1500 surplus

    def test_blatam_surplus_limited_by_remaining(self):
        """blatam needs more than available surplus -> gets partial."""
        goals = [
            _goal("רכב", preset=2000, row=50),
            _goal(BLATAM_GOAL_NAME, preset=500, outgoing=2000,
                  target=3000, total=500, row=51),
            _goal("דירה", preset=0, row=52),
        ]
        # Budget=3000, presets=2500, surplus=500
        # blatam needs: 3000 - 500 + 500 = 3000, extra = 2500, but only 500 left
        result = allocate_savings(3000, goals, default_goal_name="דירה")

        alloc_map = {a.goal_name: a for a in result.allocations}
        self.assertEqual(alloc_map[BLATAM_GOAL_NAME].allocated, 1000)  # 500 + 500
        self.assertEqual(alloc_map["דירה"].allocated, 0)  # no surplus left

    def test_blatam_no_outgoing(self):
        """blatam with outgoing=0 -> no adjustment, keeps preset."""
        goals = [
            _goal(BLATAM_GOAL_NAME, preset=500, outgoing=0,
                  target=1000, total=500, row=50),
            _goal("דירה", preset=0, row=51),
        ]
        result = allocate_savings(3000, goals, default_goal_name="דירה")

        alloc_map = {a.goal_name: a for a in result.allocations}
        self.assertEqual(alloc_map[BLATAM_GOAL_NAME].allocated, 500)
        self.assertEqual(alloc_map["דירה"].allocated, 2500)

    def test_empty_goals_list(self):
        """No goals -> empty allocations, no crash."""
        result = allocate_savings(5000, [], default_goal_name="דירה")
        self.assertEqual(len(result.allocations), 0)
        self.assertIsNone(result.warning)

    def test_exact_budget_equals_presets(self):
        """Budget exactly matches preset sum -> no surplus, no warning."""
        goals = [
            _goal("רכב", preset=2000, row=50),
            _goal("חופשה", preset=1000, row=51),
        ]
        result = allocate_savings(3000, goals, default_goal_name="רכב")

        self.assertIsNone(result.warning)
        self.assertFalse(result.zero_out)
        self.assertEqual(result.surplus, 0)
        alloc_map = {a.goal_name: a for a in result.allocations}
        self.assertEqual(alloc_map["רכב"].allocated, 2000)
        self.assertEqual(alloc_map["חופשה"].allocated, 1000)


if __name__ == "__main__":
    unittest.main()
