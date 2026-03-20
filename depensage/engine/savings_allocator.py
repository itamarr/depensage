"""
Savings section auto-allocation.

Reads the savings budget and template-preset incoming values,
then computes allocations based on three cases:
- Good month: budget >= presets → keep presets, adjust בלת"ם, surplus → default goal
- Tight month: 0 < budget < presets → keep presets, warn user
- Bad month: budget <= 0 → zero all, warn user
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
    is_blatam: bool         # True for בלת"ם goal


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
    """Read the חסכון budget line's D (budget) value.

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


def allocate_savings(budget, goals, default_goal_name=None):
    """Compute savings allocations (pure function).

    Args:
        budget: Savings budget amount (from חסכון budget line).
        goals: List of SavingsGoal from the savings section.
        default_goal_name: Name of goal that absorbs surplus. Falls back
            to first non-בלת"ם goal if not found.

    Returns:
        SavingsAllocationResult with computed allocations.
    """
    total_preset = sum(g.preset_incoming for g in goals)

    # Case 3: Bad month — budget <= 0
    if budget <= 0:
        allocations = [
            SavingsAllocation(
                goal_name=g.goal_name,
                allocated=0.0,
                row_number=g.row_number,
                is_default=(g.goal_name == default_goal_name),
                is_blatam=(g.goal_name == BLATAM_GOAL_NAME),
            )
            for g in goals
        ]
        return SavingsAllocationResult(
            allocations=allocations,
            savings_budget=budget,
            total_preset=total_preset,
            surplus=budget - total_preset,
            warning=(
                f"תקציב חסכון שלילי ({budget:,.0f}). "
                f"כל ההקצאות אופסו. מומלץ לפדות חסכונות לכיסוי הגירעון."
            ),
            zero_out=True,
        )

    # Case 2: Tight month — 0 < budget < total_preset
    if budget < total_preset:
        allocations = [
            SavingsAllocation(
                goal_name=g.goal_name,
                allocated=g.preset_incoming,
                row_number=g.row_number,
                is_default=(g.goal_name == default_goal_name),
                is_blatam=(g.goal_name == BLATAM_GOAL_NAME),
            )
            for g in goals
        ]
        return SavingsAllocationResult(
            allocations=allocations,
            savings_budget=budget,
            total_preset=total_preset,
            surplus=budget - total_preset,
            warning=(
                f"תקציב חסכון ({budget:,.0f}) נמוך מסה\"כ מתוכנן ({total_preset:,.0f}). "
                f"יש לקצץ ידנית."
            ),
            zero_out=False,
        )

    # Case 1: Good month — budget >= total_preset
    remaining = budget - total_preset

    # Find default goal
    default_goal = None
    for g in goals:
        if g.goal_name == default_goal_name:
            default_goal = g
            break
    if default_goal is None:
        # Fallback: first non-בלת"ם goal
        for g in goals:
            if g.goal_name != BLATAM_GOAL_NAME:
                default_goal = g
                break

    allocations = []
    for g in goals:
        allocated = g.preset_incoming
        is_default = (default_goal is not None and g.goal_name == default_goal.goal_name)
        is_blatam = (g.goal_name == BLATAM_GOAL_NAME)

        # Adjust בלת"ם: if outgoing > 0, increase incoming to maintain target
        if is_blatam and g.outgoing > 0:
            # Total formula: C = F(accumulated) + E(incoming) - D(outgoing)
            # We want C to stay at B (target).
            # So E = B - F + D = target - (total - incoming + outgoing) + outgoing
            # Simpler: the deficit is target - (total - incoming + outgoing)
            # incoming_needed = target - total + outgoing + incoming... no.
            # Actually: C = F + E - D where F = accumulated (carryover)
            # We want C = B (target). Current C = total.
            # If we change E to new_E: new_C = total - preset_incoming + new_E
            # We want new_C = target → new_E = target - total + preset_incoming
            needed = g.target - g.total + g.preset_incoming
            if needed > allocated:
                extra = needed - allocated
                if extra <= remaining:
                    allocated = needed
                    remaining -= extra
                else:
                    # Use whatever surplus is available
                    allocated += remaining
                    remaining = 0

        allocations.append(SavingsAllocation(
            goal_name=g.goal_name,
            allocated=allocated,
            row_number=g.row_number,
            is_default=is_default,
            is_blatam=is_blatam,
        ))

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

    Returns dict mapping goal_name → 1-based row number.
    """
    goals = read_savings_goals(handler, sheet_name)
    return {g.goal_name: g.row_number for g in goals}
