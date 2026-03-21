"""
In-memory representation of a month sheet.

Loadable from either a live sheet or the template. Provides
Python-computed equivalents of spreadsheet formulas so the
pipeline can stage everything without writing to Google Sheets.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _float(val):
    """Convert a value to float, returning 0.0 on failure."""
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


@dataclass
class BudgetLine:
    """A single budget section line."""
    category: str
    subcategory: str
    budget_amount: float    # D (preset from template)
    accumulated: float      # E (set by carryover, 0 for template)
    carry_flag: bool        # H == "CARRY"
    row_number: int         # 1-based sheet row
    remaining: float = 0.0  # B (read from sheet or computed)


@dataclass
class SavingsLine:
    """A single savings section line."""
    goal_name: str          # G
    target: float           # B
    accumulated: float      # F (set by carryover, 0 for template)
    incoming: float         # E (preset from template)
    outgoing: float         # D
    row_number: int         # 1-based sheet row


@dataclass
class VirtualMonth:
    """In-memory representation of a month sheet."""
    month: str
    year: int
    is_new: bool

    # Existing data rows (for dedup)
    expense_rows: list[list] = field(default_factory=list)   # B:G rows
    income_rows: list[list] = field(default_factory=list)    # D:G rows

    # Budget and savings structure
    budget_lines: list[BudgetLine] = field(default_factory=list)
    savings_lines: list[SavingsLine] = field(default_factory=list)

    # Section markers (1-based)
    budget_marker_row: int = 0
    income_marker_row: int = 0
    savings_marker_row: int = 0
    reconciliation_marker_row: int = 0

    # Row coordinates
    first_empty_expense_row: int = 0
    first_empty_income_row: int = 0

    # Derived values
    income_total: float | None = None
    savings_budget_value: float | None = None
    savings_budget_row: int | None = None


# ------------------------------------------------------------------
# Reading helpers
# ------------------------------------------------------------------

def _read_budget_lines(handler, sheet_name):
    """Read budget lines from a sheet's budget section.

    Returns (lines, savings_budget_value, savings_budget_row).
    """
    start_row, rows = handler.read_section_range(
        sheet_name, "budget", "B:H", end_section="income"
    )
    if not rows:
        return [], None, None

    lines = []
    savings_budget_value = None
    savings_budget_row = None

    for i, row in enumerate(rows):
        if len(row) < 6:
            continue
        category = str(row[5]).strip() if row[5] else ""
        if not category or category in ("קטגוריה", "---BUDGET---"):
            continue
        if "סה" in category and "כ" in category:
            continue

        subcategory = str(row[4]).strip() if row[4] else ""
        carry_flag = (row[6] == "CARRY") if len(row) > 6 else False
        budget_amount = _float(row[2])   # D
        accumulated = _float(row[3])     # E
        remaining = _float(row[0])       # B
        actual_row = start_row + i

        if category == "חסכון":
            savings_budget_value = budget_amount
            savings_budget_row = actual_row

        lines.append(BudgetLine(
            category=category,
            subcategory=subcategory,
            budget_amount=budget_amount,
            accumulated=accumulated,
            carry_flag=carry_flag,
            row_number=actual_row,
            remaining=remaining,
        ))

    return lines, savings_budget_value, savings_budget_row


def _read_savings_lines(handler, sheet_name):
    """Read savings lines from a sheet's savings section."""
    start_row, rows = handler.read_section_range(
        sheet_name, "savings", "A:G", end_section="reconciliation"
    )
    if not rows:
        return []

    skip_names = {"קטגוריה", 'סה"כ', "סה\"כ", "חסכון"}
    lines = []
    for i, row in enumerate(rows):
        if len(row) < 7:
            continue
        goal_name = str(row[6]).strip() if row[6] else ""
        if not goal_name or goal_name in skip_names:
            continue
        if goal_name.startswith("---"):
            continue
        if goal_name.startswith("הערה") or goal_name.startswith("העברה"):
            continue

        lines.append(SavingsLine(
            goal_name=goal_name,
            target=_float(row[1]),       # B
            accumulated=_float(row[5]),  # F
            incoming=_float(row[4]),     # E
            outgoing=_float(row[3]),     # D
            row_number=start_row + i,
        ))

    return lines


def _read_income_total(handler, sheet_name):
    """Read the income total from the income section."""
    start_row, rows = handler.read_section_range(
        sheet_name, "income", "B:G", end_section="savings"
    )
    if not rows:
        return None

    for row in rows:
        if len(row) < 5:
            continue
        for cell in row:
            if isinstance(cell, str) and "סה" in cell and "כ" in cell:
                try:
                    return float(row[3]) if len(row) > 3 and row[3] else 0.0
                except (ValueError, TypeError):
                    return 0.0
    return None


# ------------------------------------------------------------------
# Factory methods
# ------------------------------------------------------------------

def load_from_sheet(handler, month_name, year):
    """Load a VirtualMonth from an existing sheet (read-only)."""
    budget_lines, savings_budget, savings_budget_row = _read_budget_lines(
        handler, month_name
    )
    savings_lines = _read_savings_lines(handler, month_name)
    income_total = _read_income_total(handler, month_name)

    return VirtualMonth(
        month=month_name,
        year=year,
        is_new=False,
        expense_rows=handler.read_expense_rows(month_name),
        income_rows=handler.read_income_rows(month_name),
        budget_lines=budget_lines,
        savings_lines=savings_lines,
        budget_marker_row=handler.find_section_marker(month_name, "budget") or 0,
        income_marker_row=handler.find_section_marker(month_name, "income") or 0,
        savings_marker_row=handler.find_section_marker(month_name, "savings") or 0,
        reconciliation_marker_row=(
            handler.find_section_marker(month_name, "reconciliation") or 0
        ),
        first_empty_expense_row=(
            handler.find_first_empty_expense_row(month_name) or 2
        ),
        first_empty_income_row=(
            handler.find_first_empty_income_row(month_name) or 0
        ),
        income_total=income_total,
        savings_budget_value=savings_budget,
        savings_budget_row=savings_budget_row,
    )


def load_from_template(handler, month_name, year,
                       template_name="Month Template"):
    """Load a VirtualMonth from the month template (for new sheets).

    Reads the template's structure but marks the VM as new.
    Expense and income rows are empty.
    """
    budget_lines, savings_budget, savings_budget_row = _read_budget_lines(
        handler, template_name
    )
    savings_lines = _read_savings_lines(handler, template_name)

    budget_marker = (
        handler.find_section_marker(template_name, "budget") or 0
    )
    income_marker = (
        handler.find_section_marker(template_name, "income") or 0
    )
    savings_marker = (
        handler.find_section_marker(template_name, "savings") or 0
    )
    reconciliation_marker = (
        handler.find_section_marker(template_name, "reconciliation") or 0
    )

    # For a new month from template, first empty rows are at section starts
    first_empty_expense = 2  # row 2 is first expense data row
    first_empty_income = (income_marker + 3) if income_marker else 0

    return VirtualMonth(
        month=month_name,
        year=year,
        is_new=True,
        expense_rows=[],
        income_rows=[],
        budget_lines=budget_lines,
        savings_lines=savings_lines,
        budget_marker_row=budget_marker,
        income_marker_row=income_marker,
        savings_marker_row=savings_marker,
        reconciliation_marker_row=reconciliation_marker,
        first_empty_expense_row=first_empty_expense,
        first_empty_income_row=first_empty_income,
        income_total=None,
        savings_budget_value=savings_budget,
        savings_budget_row=savings_budget_row,
    )


# ------------------------------------------------------------------
# Python formula computations
# ------------------------------------------------------------------

def compute_budget_remaining(line):
    """B = D - C + E (budget - expense_sum + accumulated).

    For existing months, use line.remaining directly (already computed
    by the spreadsheet). This function is for new months where
    expense_sum is 0.
    """
    return line.budget_amount + line.accumulated


def compute_savings_total(line):
    """C = F + E - D (accumulated + incoming - outgoing)."""
    return line.accumulated + line.incoming - line.outgoing


def compute_income_total(vm):
    """Sum of income amounts from income_rows."""
    total = 0.0
    for row in vm.income_rows:
        if len(row) < 2:
            continue
        try:
            total += float(row[1]) if row[1] else 0.0
        except (ValueError, TypeError):
            pass
    return total
