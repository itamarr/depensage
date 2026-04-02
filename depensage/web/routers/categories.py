"""
Category management endpoints — read/write the Categories sheet.
"""

from fastapi import APIRouter, Depends, HTTPException

from depensage.config.settings import load_settings, get_all_years
from depensage.sheets.spreadsheet_handler import SheetHandler
from depensage.sheets.cli_helpers import fetch_categories
from depensage.web.auth import require_auth

import os

router = APIRouter(
    prefix="/api/categories", tags=["categories"],
    dependencies=[Depends(require_auth)],
)


def _get_handler():
    """Get a handler for any configured spreadsheet (categories are shared)."""
    settings = load_settings()
    credentials = os.path.abspath(settings["credentials_file"])
    # Use the first available spreadsheet
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
