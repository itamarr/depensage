"""
Category management endpoints — read/write the Categories sheet.
"""

import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from depensage.config.settings import load_settings
from depensage.sheets.spreadsheet_handler import SheetHandler
from depensage.sheets.cli_helpers import fetch_categories
from depensage.web.auth import require_auth

router = APIRouter(
    prefix="/api/categories", tags=["categories"],
    dependencies=[Depends(require_auth)],
)


def _get_handler():
    """Get a handler for any configured spreadsheet (categories are shared)."""
    settings = load_settings()
    credentials = os.path.abspath(settings["credentials_file"])
    for key, entry in settings["spreadsheets"].items():
        handler = SheetHandler(entry["id"])
        handler.authenticate(credentials)
        return handler
    raise HTTPException(status_code=500, detail="No spreadsheets configured")


@router.get("/")
async def get_categories():
    """Get all categories with their subcategories."""
    handler = _get_handler()
    cats = fetch_categories(handler)
    return {"categories": cats}


class CategoriesUpdate(BaseModel):
    categories: dict[str, list[str]]


@router.put("/")
async def update_categories(req: CategoriesUpdate):
    """Write categories back to the Categories sheet.

    Rebuilds the sheet grid: row 1 = category names, rows 2+ = subcategories.
    """
    handler = _get_handler()

    cats = req.categories
    cat_names = list(cats.keys())
    if not cat_names:
        raise HTTPException(status_code=400, detail="No categories provided")

    # Build grid: row 0 = headers, rows 1+ = subcategories
    max_subs = max((len(subs) for subs in cats.values()), default=0)
    rows = []
    rows.append(cat_names)  # Header row
    for i in range(max_subs):
        row = []
        for cat in cat_names:
            subs = cats[cat]
            row.append(subs[i] if i < len(subs) else "")
        rows.append(row)

    # Clear existing content and write new grid
    num_cols = len(cat_names)
    col_letter = chr(ord('A') + num_cols - 1) if num_cols <= 26 else 'Z'
    clear_range = f"Categories!A1:{col_letter}30"

    try:
        # Clear the range first
        handler.sheets_service.values().clear(
            spreadsheetId=handler.spreadsheet_id,
            range=clear_range,
        ).execute()

        # Write new data
        write_range = f"Categories!A1:{col_letter}{len(rows)}"
        handler.sheets_service.values().update(
            spreadsheetId=handler.spreadsheet_id,
            range=write_range,
            valueInputOption="RAW",
            body={"values": rows},
        ).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write: {e}")

    return {"status": "updated", "categories": len(cat_names)}
