"""
Month-to-month carryover logic.

Reads budget remaining and savings totals from a source month,
writes accumulated values to the destination month.
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


def _read_section_range(handler, sheet_name, section, columns, end_section=None):
    """Read a range from a section, using markers for boundaries.

    Args:
        handler: SheetHandler instance.
        sheet_name: Name of the sheet.
        section: Section name (e.g. "budget").
        columns: Column range string (e.g. "B:H").
        end_section: Next section name for end boundary. If None,
                     reads 30 rows from marker.

    Returns:
        Tuple of (start_row, list of rows) where start_row is 1-based.
        Returns (None, []) if marker not found.
    """
    start_row = handler.find_section_marker(sheet_name, section)
    if start_row is None:
        return None, []

    if end_section:
        end_row = handler.find_section_marker(sheet_name, end_section)
        if end_row is None:
            end_row = start_row + 30
    else:
        end_row = start_row + 30

    col_start, col_end = columns.split(":")
    range_str = f"{col_start}{start_row}:{col_end}{end_row - 1}"

    try:
        result = handler.sheets_service.values().get(
            spreadsheetId=handler.spreadsheet_id,
            range=f"{sheet_name}!{range_str}",
            valueRenderOption="UNFORMATTED_VALUE",
        ).execute()
        return start_row, result.get("values", [])
    except Exception as e:
        logger.error(f"Failed to read {section} section from {sheet_name}: {e}")
        return start_row, []


def _read_budget_carry_values(handler, sheet_name):
    """Read budget remaining (B) for CARRY-flagged lines.

    Returns list of dicts: {category, subcategory, remaining}.
    """
    start_row, rows = _read_section_range(
        handler, sheet_name, "budget", "B:H", end_section="income"
    )
    if not rows:
        return []

    results = []
    for row in rows:
        if len(row) < 7:
            continue
        # Columns: B=remaining, C=expense, D=budget, E=accumulated,
        #          F=subcategory, G=category, H=carry_flag
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
            # Only carry over surpluses; debts are handled by the
            # surplus/deficit formula and deducted from savings
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
    start_row, rows = _read_section_range(
        handler, sheet_name, "savings", "A:G", end_section="reconciliation"
    )
    if not rows:
        return []

    results = []
    for row in rows:
        if len(row) < 7:
            continue
        # Columns: A=months_remaining, B=target, C=total, D=outgoing,
        #          E=incoming, F=accumulated, G=goal_name
        goal = str(row[6]).strip() if row[6] else ""
        total = row[2]

        # Skip header, marker, label, and total rows
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
    start_row, rows = _read_section_range(
        handler, sheet_name, "income", "B:G", end_section="savings"
    )
    if not rows:
        return None

    # Find the total row (contains "סה"כ" in column F or G)
    for row in rows:
        if len(row) < 5:
            continue
        # Check for total label
        for cell in row:
            if isinstance(cell, str) and "סה" in cell and "כ" in cell:
                # Found total row — amount is in column E (index 3)
                try:
                    return float(row[3]) if len(row) > 3 and row[3] else 0.0
                except (ValueError, TypeError):
                    return 0.0
    return None


def _write_budget_accumulated(handler, sheet_name, carry_values):
    """Write carry-over values to budget Accumulated (E) column.

    Matches by (category, subcategory) in columns G and F.
    """
    start_row, rows = _read_section_range(
        handler, sheet_name, "budget", "B:H", end_section="income"
    )
    if not rows:
        logger.error(f"Could not read budget section in {sheet_name}")
        return 0

    # Build lookup from carry values
    carry_lookup = {
        (v["category"], v["subcategory"]): v["remaining"]
        for v in carry_values
    }

    written = 0
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
            value = carry_lookup[key]
            try:
                handler.sheets_service.values().update(
                    spreadsheetId=handler.spreadsheet_id,
                    range=f"{sheet_name}!E{actual_row}",
                    valueInputOption="USER_ENTERED",
                    body={"values": [[value]]},
                ).execute()
                written += 1
            except Exception as e:
                logger.error(
                    f"Failed to write accumulated for {category}/{subcategory}: {e}"
                )

    return written


def _write_savings_accumulated(handler, sheet_name, savings_totals):
    """Write carry-over savings totals to Accumulated (F) column.

    Matches by goal name in column G.
    """
    start_row, rows = _read_section_range(
        handler, sheet_name, "savings", "A:G", end_section="reconciliation"
    )
    if not rows:
        logger.error(f"Could not read savings section in {sheet_name}")
        return 0

    totals_lookup = {s["goal"]: s["total"] for s in savings_totals}

    written = 0
    for i, row in enumerate(rows):
        if len(row) < 7:
            continue
        goal = str(row[6]).strip() if row[6] else ""
        if goal not in totals_lookup:
            continue

        actual_row = start_row + i
        value = totals_lookup[goal]
        try:
            handler.sheets_service.values().update(
                spreadsheetId=handler.spreadsheet_id,
                range=f"{sheet_name}!F{actual_row}",
                valueInputOption="USER_ENTERED",
                body={"values": [[value]]},
            ).execute()
            written += 1
        except Exception as e:
            logger.error(f"Failed to write savings accumulated for {goal}: {e}")

    return written


def _write_savings_budget_from_income(handler, sheet_name, income_total):
    """Set the חסכון budget line so total budget = previous month's income.

    Reads all budget lines' D values (column D), finds the חסכון row,
    and sets its D value to: income_total - sum of all other budget D values.

    Returns True if written, False otherwise.
    """
    if income_total is None or income_total == 0:
        logger.info("No income total available, skipping savings budget adjustment")
        return False

    start_row, rows = _read_section_range(
        handler, sheet_name, "budget", "B:H", end_section="income"
    )
    if not rows:
        logger.error(f"Could not read budget section in {sheet_name}")
        return False

    savings_row = None
    other_budgets_sum = 0.0

    for i, row in enumerate(rows):
        if len(row) < 6:
            continue
        category = str(row[5]).strip() if row[5] else ""

        # Skip header, marker, empty, total rows
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
        logger.error(f"חסכון budget line not found in {sheet_name}")
        return False

    savings_budget = income_total - other_budgets_sum
    logger.info(
        f"Setting חסכון budget: {income_total:,.2f} - {other_budgets_sum:,.2f} "
        f"= {savings_budget:,.2f}"
    )

    try:
        handler.sheets_service.values().update(
            spreadsheetId=handler.spreadsheet_id,
            range=f"{sheet_name}!D{savings_row}",
            valueInputOption="USER_ENTERED",
            body={"values": [[savings_budget]]},
        ).execute()
        return True
    except Exception as e:
        logger.error(f"Failed to write savings budget: {e}")
        return False


def run_carryover(source_handler, source_month, dest_handler, dest_month):
    """Carry over budget and savings from source month to destination month.

    Args:
        source_handler: SheetHandler for the source spreadsheet.
        source_month: Source month sheet name (e.g. "December").
        dest_handler: SheetHandler for the destination spreadsheet.
        dest_month: Destination month sheet name (e.g. "January").

    Returns:
        Dict with keys: budget_lines, savings_lines, income_total,
        savings_budget_set.
    """
    logger.info(f"Running carryover: {source_month} → {dest_month}")

    # 1. Read source budget remaining for CARRY lines
    carry_values = _read_budget_carry_values(source_handler, source_month)
    logger.info(f"Read {len(carry_values)} CARRY budget lines from {source_month}")

    # 2. Read source savings totals
    savings_totals = _read_savings_totals(source_handler, source_month)
    logger.info(f"Read {len(savings_totals)} savings goals from {source_month}")

    # 3. Read source income total
    income_total = _read_income_total(source_handler, source_month)
    logger.info(f"Source income total: {income_total}")

    # 4. Write budget accumulated to destination
    budget_written = _write_budget_accumulated(dest_handler, dest_month, carry_values)
    logger.info(f"Wrote {budget_written} budget accumulated values to {dest_month}")

    # 5. Write savings accumulated to destination
    savings_written = _write_savings_accumulated(dest_handler, dest_month, savings_totals)
    logger.info(f"Wrote {savings_written} savings accumulated values to {dest_month}")

    # 6. Set savings budget line so total budget = source income
    savings_budget_set = _write_savings_budget_from_income(
        dest_handler, dest_month, income_total
    )

    return {
        "budget_lines": budget_written,
        "savings_lines": savings_written,
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
