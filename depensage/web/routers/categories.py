"""
Category management endpoints — read/write the Categories sheet.

Categories are read from the current year's default spreadsheet and
written to both the current year and the template spreadsheet.
"""

import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from depensage.config.settings import (
    load_settings, get_all_years, get_entries_for_year,
)
from depensage.sheets.spreadsheet_handler import SheetHandler
from depensage.sheets.cli_helpers import fetch_categories
from depensage.web.auth import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/categories", tags=["categories"],
    dependencies=[Depends(require_auth)],
)


def _get_handlers():
    """Get handler for current year + template. Returns (main_handler, template_handler_or_None)."""
    settings = load_settings()
    credentials = os.path.abspath(settings["credentials_file"])

    # Current year: latest configured year, prefer default
    years = get_all_years(settings)
    if not years:
        raise HTTPException(status_code=500, detail="No spreadsheets configured")

    current_year = max(years)
    entries = get_entries_for_year(current_year, settings)
    if not entries:
        raise HTTPException(status_code=500, detail=f"No spreadsheet for year {current_year}")

    # Prefer default entry
    main_entry = None
    for key, entry in entries:
        if entry.get("default"):
            main_entry = entry
            break
    if not main_entry:
        _, main_entry = entries[0]

    main_handler = SheetHandler(main_entry["id"])
    main_handler.authenticate(credentials)

    # Template handler (optional)
    template_handler = None
    template_entry = settings["spreadsheets"].get("template")
    if template_entry and template_entry.get("id"):
        template_handler = SheetHandler(template_entry["id"])
        template_handler.authenticate(credentials)

    return main_handler, template_handler


def _write_categories_to_sheet(handler, cats):
    """Write categories grid to a handler's Categories sheet."""
    cat_names = list(cats.keys())
    max_subs = max((len(subs) for subs in cats.values()), default=0)

    rows = [cat_names]
    for i in range(max_subs):
        row = [cats[cat][i] if i < len(cats[cat]) else "" for cat in cat_names]
        rows.append(row)

    num_cols = len(cat_names)
    col_letter = chr(ord('A') + min(num_cols - 1, 25))
    clear_range = f"Categories!A1:{col_letter}30"

    handler.sheets_service.values().clear(
        spreadsheetId=handler.spreadsheet_id,
        range=clear_range,
    ).execute()

    write_range = f"Categories!A1:{col_letter}{len(rows)}"
    handler.sheets_service.values().update(
        spreadsheetId=handler.spreadsheet_id,
        range=write_range,
        valueInputOption="RAW",
        body={"values": rows},
    ).execute()


@router.get("/")
async def get_categories():
    """Get all categories with their subcategories from the current year."""
    main_handler, _ = _get_handlers()
    cats = fetch_categories(main_handler)
    return {"categories": cats}


class CategoriesUpdate(BaseModel):
    categories: dict[str, list[str]]
    renames: dict[str, str] = {}  # old_name → new_name (for categories)
    sub_renames: dict[str, dict[str, str]] = {}  # category → {old_sub → new_sub}


@router.put("/")
async def update_categories(req: CategoriesUpdate):
    """Write categories to current year's spreadsheet and template.

    Returns rename info so the frontend can trigger propagation.
    """
    main_handler, template_handler = _get_handlers()

    cats = req.categories
    if not cats:
        raise HTTPException(status_code=400, detail="No categories provided")

    try:
        _write_categories_to_sheet(main_handler, cats)
        logger.info(f"Updated Categories sheet in main spreadsheet ({len(cats)} categories)")

        if template_handler:
            try:
                _write_categories_to_sheet(template_handler, cats)
                logger.info("Updated Categories sheet in template spreadsheet")
            except Exception as e:
                logger.warning(f"Failed to update template Categories: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write: {e}")

    return {
        "status": "updated",
        "categories": len(cats),
        "template_updated": template_handler is not None,
        "has_renames": bool(req.renames) or bool(req.sub_renames),
    }
