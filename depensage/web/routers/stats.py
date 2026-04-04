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


def _fmt_date(val):
    if val is None or val == "":
        return ""
    try:
        serial = float(val)
        dt = datetime(1899, 12, 30) + timedelta(days=serial)
        return dt.strftime("%d/%m")
    except (ValueError, TypeError):
        return str(val)


@router.get("/expenses")
async def get_expense_stats():
    """Read 'Expenses so far' pivot table data."""
    handler, year = _get_handler()

    if not handler.sheet_exists("Expenses so far"):
        return {"year": year, "rows": [], "month_count": 0}

    # Read pivot table output (A:D, up to 100 rows)
    values = handler.get_sheet_values("Expenses so far", "A1:D100")
    if not values:
        return {"year": year, "rows": [], "month_count": 0}

    # Parse: row 0 = headers, rows 1+ = data
    rows = []
    for row in values[1:]:
        if not row or len(row) < 3:
            continue
        cat = str(row[0]).strip() if row[0] else ""
        sub = str(row[1]).strip() if len(row) > 1 and row[1] else ""
        total = row[2] if len(row) > 2 else 0
        avg = row[3] if len(row) > 3 else 0

        if not cat and not sub:
            continue

        try:
            total = float(total) if total else 0
        except (ValueError, TypeError):
            total = 0
        try:
            avg = float(avg) if avg else 0
        except (ValueError, TypeError):
            avg = 0

        rows.append({
            "category": cat, "subcategory": sub,
            "total": total, "average": avg,
        })

    # Count months from Merged Expenses formula
    month_count = 0
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
                month_count = formula.count(";") + 1
        except Exception:
            pass

    return {"year": year, "rows": rows, "month_count": month_count}


@router.get("/income")
async def get_income_stats():
    """Read 'Income so far' pivot table data."""
    handler, year = _get_handler()

    if not handler.sheet_exists("Income so far"):
        return {"year": year, "rows": [], "month_count": 0}

    values = handler.get_sheet_values("Income so far", "A1:D50")
    if not values:
        return {"year": year, "rows": [], "month_count": 0}

    rows = []
    for row in values[1:]:
        if not row or len(row) < 2:
            continue
        cat = str(row[0]).strip() if row[0] else ""
        total = row[1] if len(row) > 1 else 0
        avg = row[2] if len(row) > 2 else 0

        if not cat:
            continue

        try:
            total = float(total) if total else 0
        except (ValueError, TypeError):
            total = 0
        try:
            avg = float(avg) if avg else 0
        except (ValueError, TypeError):
            avg = 0

        rows.append({"category": cat, "total": total, "average": avg})

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

        # Total expenses from expense rows
        expense_total = 0.0
        for row in vm.expense_rows:
            try:
                expense_total += float(row[3]) if len(row) > 3 and row[3] else 0
            except (ValueError, TypeError):
                pass

        # Income total
        income_total = vm.income_total or 0.0

        # Savings budget
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

    # Find the latest month with data
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
