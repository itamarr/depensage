"""
Month data viewing and editing endpoints.
"""

import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from depensage.config.settings import (
    load_settings, get_spreadsheet_entry, get_entries_for_year, get_all_years,
)
from depensage.sheets.spreadsheet_handler import SheetHandler
from depensage.engine.virtual_month import (
    load_from_sheet, BudgetLine, SavingsLine, _read_income_total,
)
from depensage.web.auth import require_auth

router = APIRouter(
    prefix="/api/months", tags=["months"],
    dependencies=[Depends(require_auth)],
)


def _get_handler(year: int):
    settings = load_settings()
    credentials = os.path.abspath(settings["credentials_file"])
    entries = get_entries_for_year(year, settings)
    if not entries:
        raise HTTPException(status_code=404, detail=f"No spreadsheet for year {year}")
    # Prefer default
    for key, entry in entries:
        if entry.get("default"):
            h = SheetHandler(entry["id"])
            h.authenticate(credentials)
            return h, entry
    _, entry = entries[0]
    h = SheetHandler(entry["id"])
    h.authenticate(credentials)
    return h, entry


MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


@router.get("/")
async def list_months():
    """List available months across all years."""
    years = get_all_years()
    result = []
    for year in years:
        try:
            handler, entry = _get_handler(year)
        except HTTPException:
            continue
        for month in MONTH_NAMES:
            if handler.sheet_exists(month):
                result.append({"month": month, "year": year})
    return {"months": result}


@router.get("/{year}/{month}/expenses")
async def get_expenses(year: int, month: str):
    """Get expense rows for a month."""
    handler, entry = _get_handler(year)
    if not handler.sheet_exists(month):
        raise HTTPException(status_code=404, detail=f"{month} {year} not found")

    rows = handler.read_expense_rows_with_status(month)
    expenses = []
    for i, row in enumerate(rows):
        if not row or len(row) < 6:
            continue
        # Skip empty rows
        biz = str(row[0]).strip() if row[0] else ""
        amount = row[3] if len(row) > 3 else ""
        if not biz and not amount:
            continue
        expenses.append({
            "index": i,
            "business_name": biz,
            "notes": str(row[1]).strip() if len(row) > 1 and row[1] else "",
            "subcategory": str(row[2]).strip() if len(row) > 2 and row[2] else "",
            "amount": str(row[3]) if len(row) > 3 and row[3] else "",
            "category": str(row[4]).strip() if len(row) > 4 and row[4] else "",
            "date": str(row[5]) if len(row) > 5 else "",
            "status": str(row[6]).strip() if len(row) > 6 and row[6] else "",
        })
    return {"month": month, "year": year, "expenses": expenses}


@router.get("/{year}/{month}/budget")
async def get_budget(year: int, month: str):
    """Get budget lines and savings lines for a month."""
    handler, entry = _get_handler(year)
    if not handler.sheet_exists(month):
        raise HTTPException(status_code=404, detail=f"{month} {year} not found")

    vm = load_from_sheet(handler, month, year)

    budget_lines = [
        {
            "category": bl.category,
            "subcategory": bl.subcategory,
            "budget_amount": bl.budget_amount,
            "accumulated": bl.accumulated,
            "remaining": bl.remaining,
            "carry_flag": bl.carry_flag,
            "row_number": bl.row_number,
        }
        for bl in vm.budget_lines
    ]

    savings_lines = [
        {
            "goal_name": sl.goal_name,
            "target": sl.target,
            "accumulated": sl.accumulated,
            "incoming": sl.incoming,
            "outgoing": sl.outgoing,
            "total": sl.accumulated + sl.incoming - sl.outgoing,
            "row_number": sl.row_number,
        }
        for sl in vm.savings_lines
    ]

    return {
        "month": month, "year": year,
        "budget_lines": budget_lines,
        "savings_lines": savings_lines,
        "savings_budget": vm.savings_budget_value,
        "income_total": vm.income_total,
    }


@router.get("/{year}/{month}/income")
async def get_income(year: int, month: str):
    """Get income rows and reconciliation data for a month."""
    handler, entry = _get_handler(year)
    if not handler.sheet_exists(month):
        raise HTTPException(status_code=404, detail=f"{month} {year} not found")

    rows = handler.read_income_rows(month)
    income = []
    for i, row in enumerate(rows):
        if not row or len(row) < 2:
            continue
        amount = row[1] if len(row) > 1 else ""
        if not amount:
            continue
        income.append({
            "index": i,
            "comments": str(row[0]).strip() if row[0] else "",
            "amount": str(row[1]) if len(row) > 1 else "",
            "category": str(row[2]).strip() if len(row) > 2 and row[2] else "",
            "date": str(row[3]) if len(row) > 3 else "",
        })

    # Reconciliation: read key cells
    recon = {}
    for label in ['כסף בעו"ש', 'פער']:
        row_num = handler.find_reconciliation_label_row(month, label)
        if row_num:
            recon[label] = {"row": row_num}

    return {
        "month": month, "year": year,
        "income": income,
        "reconciliation": recon,
    }


@router.get("/{year}/{month}/link")
async def get_sheet_link(year: int, month: str):
    """Get a direct Google Sheets URL for this month."""
    handler, entry = _get_handler(year)
    sheet_id = handler.get_sheet_id(month)
    spreadsheet_id = entry["id"]
    if sheet_id is not None:
        url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit#gid={sheet_id}"
    else:
        url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
    return {"url": url}


class CellEdit(BaseModel):
    cell_ref: str  # e.g. "F5", "E120"
    value: str | float


class BatchEditRequest(BaseModel):
    edits: list[CellEdit]


@router.put("/{year}/{month}/cells")
async def edit_cells(year: int, month: str, req: BatchEditRequest):
    """Write cells directly to a month sheet (for corrections)."""
    handler, entry = _get_handler(year)
    if not handler.sheet_exists(month):
        raise HTTPException(status_code=404, detail=f"{month} {year} not found")

    updates = [(e.cell_ref, e.value) for e in req.edits]
    success = handler.batch_update_cells(month, updates)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to write cells")

    return {"status": "updated", "count": len(updates)}
