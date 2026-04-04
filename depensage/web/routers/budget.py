"""
Budget planning endpoints — read/write the Month Template's budget section.
"""

import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from depensage.config.settings import load_settings, get_all_years, get_entries_for_year
from depensage.sheets.spreadsheet_handler import SheetHandler
from depensage.engine.virtual_month import load_from_sheet
from depensage.web.auth import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/budget", tags=["budget"],
    dependencies=[Depends(require_auth)],
)


def _get_handlers():
    """Get handler for current year + template spreadsheet."""
    settings = load_settings()
    credentials = os.path.abspath(settings["credentials_file"])
    years = get_all_years(settings)
    if not years:
        raise HTTPException(status_code=500, detail="No spreadsheets configured")

    current_year = max(years)
    entries = get_entries_for_year(current_year, settings)
    if not entries:
        raise HTTPException(status_code=500, detail=f"No spreadsheet for {current_year}")

    main_entry = None
    for key, entry in entries:
        if entry.get("default"):
            main_entry = entry
            break
    if not main_entry:
        _, main_entry = entries[0]

    main_handler = SheetHandler(main_entry["id"])
    main_handler.authenticate(credentials)

    template_handler = None
    template_entry = settings["spreadsheets"].get("template")
    if template_entry and template_entry.get("id"):
        template_handler = SheetHandler(template_entry["id"])
        template_handler.authenticate(credentials)

    return main_handler, template_handler, current_year


def _parse_num(val) -> float:
    if val is None or val == "":
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).replace("₪", "").replace(",", "").replace(" ", "").strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


@router.get("/")
async def get_budget():
    """Read the Month Template's budget section with avg spent context."""
    main_handler, _, year = _get_handlers()

    if not main_handler.sheet_exists("Month Template"):
        raise HTTPException(status_code=404, detail="Month Template not found")

    vm = load_from_sheet(main_handler, "Month Template", year)

    lines = []
    for bl in vm.budget_lines:
        lines.append({
            "category": bl.category,
            "subcategory": bl.subcategory,
            "budget_amount": bl.budget_amount,
            "carry_status": bl.carry_status,
            "row_number": bl.row_number,
        })

    # Get averages from "Expenses so far" pivot for context
    avg_by_key = {}
    if main_handler.sheet_exists("Expenses so far"):
        values = main_handler.get_sheet_values("Expenses so far", "A1:D100")
        if values:
            for row in values:
                if not row or len(row) < 4:
                    continue
                cat = str(row[0]).strip() if row[0] else ""
                sub = str(row[1]).strip() if len(row) > 1 and row[1] else ""
                avg = _parse_num(row[3] if len(row) > 3 else 0)
                if "Total" in cat:
                    avg_by_key[cat.replace(" Total", "")] = avg
                elif cat and sub:
                    avg_by_key[f"{cat}/{sub}"] = avg
                elif cat and not sub:
                    avg_by_key[cat] = avg

    # Count months
    month_count = 0
    if main_handler.sheet_exists("Merged Expenses"):
        try:
            result = main_handler.sheets_service.values().get(
                spreadsheetId=main_handler.spreadsheet_id,
                range="'Merged Expenses'!A2",
                valueRenderOption="FORMULA",
            ).execute()
            formula_vals = result.get("values", [])
            if formula_vals and formula_vals[0]:
                month_count = str(formula_vals[0][0]).count(":")
        except Exception:
            pass

    return {
        "year": year,
        "lines": lines,
        "averages": avg_by_key,
        "month_count": month_count,
    }


class BudgetLineUpdate(BaseModel):
    row_number: int
    budget_amount: float


class BudgetSaveRequest(BaseModel):
    updates: list[BudgetLineUpdate]


@router.put("/")
async def save_budget(req: BudgetSaveRequest):
    """Write budget amounts to both Month Template sheets."""
    main_handler, template_handler, year = _get_handlers()

    if not req.updates:
        return {"status": "no changes"}

    # Build cell updates: column D (budget amount) for each row
    cell_updates = [(f"D{u.row_number}", u.budget_amount) for u in req.updates]

    # Write to current year's Month Template
    main_handler.invalidate_cache("Month Template")
    success = main_handler.batch_update_cells("Month Template", cell_updates)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to write to Month Template")

    logger.info(f"Updated {len(cell_updates)} budget lines in Month Template")

    # Write to template spreadsheet
    template_updated = False
    if template_handler and template_handler.sheet_exists("Month Template"):
        try:
            template_handler.batch_update_cells("Month Template", cell_updates)
            template_updated = True
            logger.info("Updated budget in template spreadsheet")
        except Exception as e:
            logger.warning(f"Failed to update template spreadsheet: {e}")

    return {
        "status": "updated",
        "lines_updated": len(cell_updates),
        "template_updated": template_updated,
    }
