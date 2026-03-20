"""
Savings section auto-allocation.

Reads the savings budget and template-preset incoming values,
then computes allocations based on three cases:
- Good month: budget >= presets -> keep presets, adjust blatam, surplus -> default goal
- Tight month: 0 < budget < presets -> keep presets, warn user
- Bad month: budget <= 0 -> zero all, warn user
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

BLATAM_GOAL_NAME = 'בלת"ם'

# Skip rows that aren't actual savings goals
SKIP_GOAL_NAMES = {"קטגוריה", "סה\"כ", 'סה"כ', "חסכון"}


@dataclass
class SavingsGoal:
    goal_name: str
    preset_incoming: float  # column E (template preset)
    outgoing: float         # column D
    target: float           # column B
    total: float            # column C
    row_number: int         # 1-based sheet row


@dataclass
class SavingsAllocation:
    goal_name: str
    allocated: float        # what to write to column E
    row_number: int
    is_default: bool        # True for the surplus-absorbing goal
    is_blatam: bool         # True for blatam goal
    # Context fields for XLSX display
    preset_incoming: float = 0.0
    target: float = 0.0
    current_total: float = 0.0


@dataclass
class SavingsAllocationResult:
    allocations: list[SavingsAllocation]
    savings_budget: float
    total_preset: float         # sum of template presets
    surplus: float              # budget - total_preset (before adjustments)
    warning: str | None         # set for case 2 and 3
    zero_out: bool              # True for case 3


def _is_skip_goal(name):
    """Check if a goal name should be skipped (header, marker, total, etc.)."""
    if not name:
        return True
    if name in SKIP_GOAL_NAMES:
        return True
    if name.startswith("---"):
        return True
    if name.startswith("הערה") or name.startswith("העברה"):
        return True
    return False


def read_savings_goals(handler, sheet_name):
    """Read savings goals between savings and reconciliation markers.

    Columns: A=months_remaining, B=target, C=total, D=outgoing,
             E=incoming, F=accumulated, G=goal_name.

    Returns list of SavingsGoal.
    """
    start_row, rows = handler.read_section_range(
        sheet_name, "savings", "A:G", end_section="reconciliation"
    )
    if not rows:
        return []

    goals = []
    for i, row in enumerate(rows):
        if len(row) < 7:
            continue
        goal_name = str(row[6]).strip() if row[6] else ""
        if _is_skip_goal(goal_name):
            continue

        def _float(val):
            if val is None:
                return 0.0
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0

        goals.append(SavingsGoal(
            goal_name=goal_name,
            preset_incoming=_float(row[4]),  # E = incoming
            outgoing=_float(row[3]),          # D = outgoing
            target=_float(row[1]),            # B = target
            total=_float(row[2]),             # C = total
            row_number=start_row + i,
        ))

    return goals


def read_savings_budget(handler, sheet_name):
    """Read the savings budget line's D (budget) value.

    Returns float or None if not found.
    """
    start_row, rows = handler.read_section_range(
        sheet_name, "budget", "B:H", end_section="income"
    )
    if not rows:
        return None

    for row in rows:
        if len(row) < 6:
            continue
        category = str(row[5]).strip() if row[5] else ""
        if category == "חסכון":
            budget_val = row[2] if len(row) > 2 else None  # D column
            if budget_val is not None:
                try:
                    return float(budget_val)
                except (ValueError, TypeError):
                    return None
            return None

    return None


def _make_allocation(g, allocated, default_goal_name, default_goal):
    """Create a SavingsAllocation with context fields from a SavingsGoal."""
    is_default = (default_goal is not None and g.goal_name == default_goal.goal_name)
    return SavingsAllocation(
        goal_name=g.goal_name,
        allocated=allocated,
        row_number=g.row_number,
        is_default=is_default,
        is_blatam=(g.goal_name == BLATAM_GOAL_NAME),
        preset_incoming=g.preset_incoming,
        target=g.target,
        current_total=g.total,
    )


def allocate_savings(budget, goals, default_goal_name=None):
    """Compute savings allocations (pure function).

    Args:
        budget: Savings budget amount (from savings budget line).
        goals: List of SavingsGoal from the savings section.
        default_goal_name: Name of goal that absorbs surplus. Falls back
            to first non-blatam goal if not found.

    Returns:
        SavingsAllocationResult with computed allocations.
    """
    total_preset = sum(g.preset_incoming for g in goals)

    # Find default goal (needed for all cases)
    default_goal = None
    for g in goals:
        if g.goal_name == default_goal_name:
            default_goal = g
            break
    if default_goal is None:
        for g in goals:
            if g.goal_name != BLATAM_GOAL_NAME:
                default_goal = g
                break

    # Case 3: Bad month — budget <= 0
    if budget <= 0:
        allocations = [
            _make_allocation(g, 0.0, default_goal_name, default_goal)
            for g in goals
        ]
        return SavingsAllocationResult(
            allocations=allocations,
            savings_budget=budget,
            total_preset=total_preset,
            surplus=budget - total_preset,
            warning=(
                f"Negative savings budget ({budget:,.0f}). "
                f"All allocations zeroed. Consider liquidating savings to cover the deficit."
            ),
            zero_out=True,
        )

    # Case 2: Tight month — 0 < budget < total_preset
    if budget < total_preset:
        allocations = [
            _make_allocation(g, g.preset_incoming, default_goal_name, default_goal)
            for g in goals
        ]
        return SavingsAllocationResult(
            allocations=allocations,
            savings_budget=budget,
            total_preset=total_preset,
            surplus=budget - total_preset,
            warning=(
                f"Savings budget ({budget:,.0f}) is less than total planned "
                f"({total_preset:,.0f}). Adjust allocations manually."
            ),
            zero_out=False,
        )

    # Case 1: Good month — budget >= total_preset
    remaining = budget - total_preset

    allocations = []
    for g in goals:
        allocated = g.preset_incoming
        is_blatam = (g.goal_name == BLATAM_GOAL_NAME)

        # Adjust blatam: if outgoing > 0, increase incoming to maintain target
        if is_blatam and g.outgoing > 0:
            # C = F + E - D. We want C = B (target).
            # If we change E to new_E: new_C = total - preset_incoming + new_E
            # We want new_C = target -> new_E = target - total + preset_incoming
            needed = g.target - g.total + g.preset_incoming
            if needed > allocated:
                extra = needed - allocated
                if extra <= remaining:
                    allocated = needed
                    remaining -= extra
                else:
                    allocated += remaining
                    remaining = 0

        allocations.append(
            _make_allocation(g, allocated, default_goal_name, default_goal)
        )

    # Put remaining surplus in default goal
    if remaining > 0 and default_goal is not None:
        for a in allocations:
            if a.is_default:
                a.allocated += remaining
                break

    return SavingsAllocationResult(
        allocations=allocations,
        savings_budget=budget,
        total_preset=total_preset,
        surplus=budget - total_preset,
        warning=None,
        zero_out=False,
    )


def find_savings_goal_rows(handler, sheet_name):
    """Re-scan savings section to get fresh row numbers by goal name.

    Returns dict mapping goal_name -> 1-based row number.
    """
    goals = read_savings_goals(handler, sheet_name)
    return {g.goal_name: g.row_number for g in goals}
