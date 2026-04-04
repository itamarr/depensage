"""
Statistics endpoints — reads from pivot sheets and month data.
"""

import os
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException

from depensage.config.settings import (
    load_settings, get_all_years, get_entries_for_year,
)
from depensage.sheets.spreadsheet_handler import SheetHandler
from depensage.engine.virtual_month import load_from_sheet
from depensage.web.auth import require_auth

router = APIRouter(
    prefix="/api/stats", tags=["stats"],
    dependencies=[Depends(require_auth)],
)

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _get_handler():
    settings = load_settings()
    credentials = os.path.abspath(settings["credentials_file"])
    years = get_all_years(settings)
    if not years:
        raise HTTPException(status_code=500, detail="No spreadsheets configured")
    current_year = max(years)
    entries = get_entries_for_year(current_year, settings)
    for key, entry in entries:
        if entry.get("default"):
            h = SheetHandler(entry["id"])
            h.authenticate(credentials)
            return h, current_year
    _, entry = entries[0]
    h = SheetHandler(entry["id"])
    h.authenticate(credentials)
    return h, current_year


def _parse_num(val) -> float:
    """Parse a number that may be NIS-formatted (e.g., '2,664.45 ₪')."""
    if val is None or val == "":
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).replace("₪", "").replace(",", "").replace(" ", "").strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


@router.get("/expenses")
async def get_expense_stats():
    """Read 'Expenses so far' pivot table + budget from latest month."""
    handler, year = _get_handler()

    # Read pivot table
    pivot_rows = []
    month_count = 0

    if handler.sheet_exists("Expenses so far"):
        values = handler.get_sheet_values("Expenses so far", "A1:D100")
        if values:
            for row in values:
                if not row or len(row) < 3:
                    continue
                cat = str(row[0]).strip() if row[0] else ""
                sub = str(row[1]).strip() if len(row) > 1 and row[1] else ""
                total = _parse_num(row[2] if len(row) > 2 else 0)
                avg = _parse_num(row[3] if len(row) > 3 else 0)

                if not cat and not sub:
                    continue

                is_total = "Total" in cat
                is_grand = cat == "Grand Total"

                pivot_rows.append({
                    "category": cat.replace(" Total", "") if is_total else cat,
                    "subcategory": sub,
                    "total": total, "average": avg,
                    "is_total": is_total, "is_grand": is_grand,
                })

    # Count months from Merged Expenses formula
    if handler.sheet_exists("Merged Expenses"):
        try:
            result = handler.sheets_service.values().get(
                spreadsheetId=handler.spreadsheet_id,
                range="'Merged Expenses'!A2",
                valueRenderOption="FORMULA",
            ).execute()
            formula_vals = result.get("values", [])
            if formula_vals and formula_vals[0]:
                formula = str(formula_vals[0][0])
                month_count = formula.count(":")
        except Exception:
            pass

    # Read budget from latest month
    budget_by_cat = {}
    for month in reversed(MONTH_NAMES):
        if not handler.sheet_exists(month):
            continue
        vm = load_from_sheet(handler, month, year)
        if not vm.budget_lines:
            continue
        for bl in vm.budget_lines:
            # For שונות (misc), budget is per subcategory
            if bl.subcategory:
                key = f"{bl.category}/{bl.subcategory}"
            else:
                key = bl.category
            budget_by_cat[key] = bl.budget_amount
        break  # Only need the latest month

    return {
        "year": year, "rows": pivot_rows,
        "month_count": month_count, "budget": budget_by_cat,
    }


@router.get("/income")
async def get_income_stats():
    """Read 'Income so far' pivot table data."""
    handler, year = _get_handler()

    if not handler.sheet_exists("Income so far"):
        return {"year": year, "rows": []}

    # Income so far: A=category, B=details, C=total, D=average
    values = handler.get_sheet_values("Income so far", "A1:D50")
    if not values:
        return {"year": year, "rows": []}

    rows = []
    for row in values:
        if not row or len(row) < 1:
            continue
        cat = str(row[0]).strip() if row[0] else ""
        if not cat:
            continue

        is_total = "Total" in cat
        is_grand = cat == "Grand Total"
        details = str(row[1]).strip() if len(row) > 1 and row[1] else ""
        total = _parse_num(row[2] if len(row) > 2 else 0)
        avg = _parse_num(row[3] if len(row) > 3 else 0)

        # Skip "Total" duplicate rows — they just repeat the category total
        if is_total and not is_grand:
            continue

        rows.append({
            "category": cat, "details": details,
            "total": total, "average": avg,
            "is_grand": is_grand,
        })

    return {"year": year, "rows": rows}


@router.get("/monthly")
async def get_monthly_totals():
    """Get per-month expense and income totals for trend charts."""
    handler, year = _get_handler()

    months = []
    for month in MONTH_NAMES:
        if not handler.sheet_exists(month):
            continue

        vm = load_from_sheet(handler, month, year)

        expense_total = 0.0
        for row in vm.expense_rows:
            try:
                expense_total += float(row[3]) if len(row) > 3 and row[3] else 0
            except (ValueError, TypeError):
                pass

        income_total = vm.income_total or 0.0
        savings_budget = vm.savings_budget_value or 0.0

        months.append({
            "month": month,
            "expenses": round(expense_total, 2),
            "income": round(income_total, 2),
            "savings_budget": round(savings_budget, 2),
        })

    return {"year": year, "months": months}


@router.get("/savings")
async def get_savings_overview():
    """Get current savings goal status from the latest month."""
    handler, year = _get_handler()

    for month in reversed(MONTH_NAMES):
        if not handler.sheet_exists(month):
            continue

        vm = load_from_sheet(handler, month, year)
        if not vm.savings_lines:
            continue

        goals = []
        for sl in vm.savings_lines:
            total = sl.accumulated + sl.incoming - sl.outgoing
            goals.append({
                "goal_name": sl.goal_name,
                "target": sl.target,
                "total": round(total, 2),
                "progress": round(total / sl.target * 100, 1) if sl.target > 0 else 0,
            })

        return {"year": year, "month": month, "goals": goals}

    return {"year": year, "month": None, "goals": []}
