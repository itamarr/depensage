"""
Month-to-month carryover logic.

Reads budget remaining and savings totals from a source month,
writes accumulated values to the destination month.
All writes are batched to minimize API calls.
"""

import logging

logger = logging.getLogger(__name__)

PREV_MONTH = {
    "January": ("December", -1),    # previous year
    "February": ("January", 0),
    "March": ("February", 0),
    "April": ("March", 0),
    "May": ("April", 0),
    "June": ("May", 0),
    "July": ("June", 0),
    "August": ("July", 0),
    "September": ("August", 0),
    "October": ("September", 0),
    "November": ("October", 0),
    "December": ("November", 0),
}


def _read_budget_carry_values(handler, sheet_name):
    """Read budget remaining (B) for CARRY-flagged lines.

    Returns list of dicts: {category, subcategory, remaining}.
    """
    start_row, rows = handler.read_section_range(
        sheet_name, "budget", "B:H", end_section="income"
    )
    if not rows:
        return []

    results = []
    for row in rows:
        if len(row) < 7:
            continue
        carry_flag = row[6] if len(row) > 6 else ""
        if carry_flag != "CARRY":
            continue

        category = str(row[5]).strip() if row[5] else ""
        subcategory = str(row[4]).strip() if row[4] else ""
        remaining = row[0]

        if category and remaining is not None:
            try:
                remaining = float(remaining)
            except (ValueError, TypeError):
                continue
            if remaining > 0:
                results.append({
                    "category": category,
                    "subcategory": subcategory,
                    "remaining": remaining,
                })

    return results


def _read_savings_totals(handler, sheet_name):
    """Read savings Total (C) for each goal.

    Returns list of dicts: {goal, total}.
    """
    start_row, rows = handler.read_section_range(
        sheet_name, "savings", "A:G", end_section="reconciliation"
    )
    if not rows:
        return []

    results = []
    for row in rows:
        if len(row) < 7:
            continue
        goal = str(row[6]).strip() if row[6] else ""
        total = row[2]

        if not goal or goal in ("קטגוריה", "סה\"כ", 'סה"כ', "חסכון"):
            continue
        if goal.startswith("---"):
            continue
        if goal.startswith("הערה") or goal.startswith("העברה"):
            continue

        if total is not None:
            try:
                total = float(total)
            except (ValueError, TypeError):
                total = 0.0
        else:
            total = 0.0

        results.append({"goal": goal, "total": total})

    return results


def _read_income_total(handler, sheet_name):
    """Read the income total from the income section.

    Returns float or None.
    """
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


def _collect_budget_carry_formulas(handler, dest_month, source_month):
    """Collect carry-over formula updates for budget Accumulated (E) column.

    Returns list of (cell_ref, formula) tuples.
    """
    start_row, rows = handler.read_section_range(
        dest_month, "budget", "B:H", end_section="income"
    )
    if not rows:
        logger.error(f"Could not read budget section in {dest_month}")
        return []

    updates = []
    for i, row in enumerate(rows):
        if len(row) < 7:
            continue
        carry_flag = row[6] if len(row) > 6 else ""
        if carry_flag != "CARRY":
            continue

        actual_row = start_row + i
        formula = f"=MAX('{source_month}'!B{actual_row},0)"
        updates.append((f"E{actual_row}", formula))

    return updates


def _collect_budget_accumulated_static(handler, sheet_name, carry_values):
    """Collect static carry-over updates for budget Accumulated (E) column.

    Used for cross-year carryover. Returns list of (cell_ref, value) tuples.
    """
    start_row, rows = handler.read_section_range(
        sheet_name, "budget", "B:H", end_section="income"
    )
    if not rows:
        logger.error(f"Could not read budget section in {sheet_name}")
        return []

    carry_lookup = {
        (v["category"], v["subcategory"]): v["remaining"]
        for v in carry_values
    }

    updates = []
    for i, row in enumerate(rows):
        if len(row) < 7:
            continue
        carry_flag = row[6] if len(row) > 6 else ""
        if carry_flag != "CARRY":
            continue

        category = str(row[5]).strip() if row[5] else ""
        subcategory = str(row[4]).strip() if row[4] else ""
        key = (category, subcategory)

        if key in carry_lookup:
            actual_row = start_row + i
            updates.append((f"E{actual_row}", carry_lookup[key]))

    return updates


def _collect_savings_carry_formulas(handler, dest_month, source_month):
    """Collect carry-over formula updates for savings Accumulated (F) column.

    Returns list of (cell_ref, formula) tuples.
    """
    start_row, rows = handler.read_section_range(
        dest_month, "savings", "A:G", end_section="reconciliation"
    )
    if not rows:
        logger.error(f"Could not read savings section in {dest_month}")
        return []

    updates = []
    for i, row in enumerate(rows):
        if len(row) < 7:
            continue
        goal = str(row[6]).strip() if row[6] else ""

        if not goal or goal in ("קטגוריה", "סה\"כ", 'סה"כ', "חסכון"):
            continue
        if goal.startswith("---") or goal.startswith("הערה") or goal.startswith("העברה"):
            continue

        actual_row = start_row + i
        formula = f"='{source_month}'!C{actual_row}"
        updates.append((f"F{actual_row}", formula))

    return updates


def _collect_savings_accumulated_static(handler, sheet_name, savings_totals):
    """Collect static savings carry-over updates for Accumulated (F) column.

    Returns list of (cell_ref, value) tuples.
    """
    start_row, rows = handler.read_section_range(
        sheet_name, "savings", "A:G", end_section="reconciliation"
    )
    if not rows:
        logger.error(f"Could not read savings section in {sheet_name}")
        return []

    totals_lookup = {s["goal"]: s["total"] for s in savings_totals}

    updates = []
    for i, row in enumerate(rows):
        if len(row) < 7:
            continue
        goal = str(row[6]).strip() if row[6] else ""
        if goal not in totals_lookup:
            continue

        actual_row = start_row + i
        updates.append((f"F{actual_row}", totals_lookup[goal]))

    return updates


def _collect_savings_budget_update(handler, sheet_name, income_total):
    """Compute savings budget and return (cell_ref, value) or None.

    Sets the savings budget line D = income_total - sum(other budget D values).
    """
    if income_total is None or income_total == 0:
        logger.info("No income total available, skipping savings budget adjustment")
        return None

    start_row, rows = handler.read_section_range(
        sheet_name, "budget", "B:H", end_section="income"
    )
    if not rows:
        logger.error(f"Could not read budget section in {sheet_name}")
        return None

    savings_row = None
    other_budgets_sum = 0.0

    for i, row in enumerate(rows):
        if len(row) < 6:
            continue
        category = str(row[5]).strip() if row[5] else ""

        if not category or category in ("קטגוריה", "---BUDGET---"):
            continue
        if "סה" in category and "כ" in category:
            continue

        budget_val = row[2] if len(row) > 2 else 0
        try:
            budget_val = float(budget_val) if budget_val else 0.0
        except (ValueError, TypeError):
            budget_val = 0.0

        if category == "חסכון":
            savings_row = start_row + i
        else:
            other_budgets_sum += budget_val

    if savings_row is None:
        logger.error(f"Savings budget line not found in {sheet_name}")
        return None

    savings_budget = income_total - other_budgets_sum
    logger.info(
        f"Setting savings budget: {income_total:,.2f} - {other_budgets_sum:,.2f} "
        f"= {savings_budget:,.2f}"
    )

    return (f"D{savings_row}", savings_budget)


def run_carryover(source_handler, source_month, dest_handler, dest_month):
    """Carry over budget and savings from source month to destination month.

    All cell updates are batched into a single API call.

    Args:
        source_handler: SheetHandler for the source spreadsheet.
        source_month: Source month sheet name (e.g. "December").
        dest_handler: SheetHandler for the destination spreadsheet.
        dest_month: Destination month sheet name (e.g. "January").

    Returns:
        Dict with keys: budget_lines, savings_lines, income_total,
        savings_budget_set.
    """
    logger.info(f"Running carryover: {source_month} -> {dest_month}")

    same_spreadsheet = (
        hasattr(source_handler, 'spreadsheet_id')
        and hasattr(dest_handler, 'spreadsheet_id')
        and source_handler.spreadsheet_id == dest_handler.spreadsheet_id
    )

    all_updates = []  # (cell_ref, value) pairs for dest sheet

    # 1. Budget carry
    if same_spreadsheet:
        budget_updates = _collect_budget_carry_formulas(
            dest_handler, dest_month, source_month
        )
        logger.info(
            f"Collected {len(budget_updates)} budget carry formulas "
            f"(={source_month}!B...) for {dest_month}"
        )
    else:
        carry_values = _read_budget_carry_values(source_handler, source_month)
        logger.info(f"Read {len(carry_values)} CARRY budget lines from {source_month}")
        budget_updates = _collect_budget_accumulated_static(
            dest_handler, dest_month, carry_values
        )
    all_updates.extend(budget_updates)

    # 2. Savings carry
    if same_spreadsheet:
        savings_updates = _collect_savings_carry_formulas(
            dest_handler, dest_month, source_month
        )
        logger.info(
            f"Collected {len(savings_updates)} savings carry formulas "
            f"(='{source_month}'!C...) for {dest_month}"
        )
    else:
        savings_totals = _read_savings_totals(source_handler, source_month)
        logger.info(f"Read {len(savings_totals)} savings goals from {source_month}")
        savings_updates = _collect_savings_accumulated_static(
            dest_handler, dest_month, savings_totals
        )
    all_updates.extend(savings_updates)

    # 3. Read source income total and set savings budget
    income_total = _read_income_total(source_handler, source_month)
    logger.info(f"Source income total: {income_total}")

    budget_update = _collect_savings_budget_update(
        dest_handler, dest_month, income_total
    )
    savings_budget_set = budget_update is not None
    if budget_update:
        all_updates.append(budget_update)

    # 4. Batch write all updates
    if all_updates:
        dest_handler.batch_update_cells(dest_month, all_updates)
        logger.info(f"Wrote {len(all_updates)} carryover updates to {dest_month}")

    return {
        "budget_lines": len(budget_updates),
        "savings_lines": len(savings_updates),
        "income_total": income_total,
        "savings_budget_set": savings_budget_set,
    }


def get_previous_month(month_name):
    """Get the previous month name and year offset.

    Args:
        month_name: Current month name (e.g. "January").

    Returns:
        Tuple of (prev_month_name, year_offset) where year_offset
        is -1 for January (previous year) or 0 otherwise.
        Returns (None, 0) if month_name is invalid.
    """
    entry = PREV_MONTH.get(month_name)
    if entry is None:
        return None, 0
    return entry
