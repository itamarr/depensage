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


class PropagateRequest(BaseModel):
    renames: dict[str, str] = {}  # old_cat → new_cat
    sub_renames: dict[str, dict[str, str]] = {}  # category → {old_sub → new_sub}
    removals: list[str] = []  # removed categories
    removal_replacements: dict[str, str] = {}  # removed_cat → replacement_cat


@router.post("/propagate")
async def propagate_renames(req: PropagateRequest):
    """Propagate category/subcategory renames across all months in the current year.

    Updates:
    - Expense rows: column F (category), column D (subcategory)
    - Budget section: column G (category), column F (subcategory)
    - Lookup tables: CC, bank, income classifiers
    """
    main_handler, _ = _get_handlers()
    settings = load_settings()

    # Get current year
    years = get_all_years(settings)
    current_year = max(years)

    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]

    total_updates = 0

    for month in month_names:
        if not main_handler.sheet_exists(month):
            continue

        updates = []

        # Read expense rows (B:H = columns 1-7, 0-indexed in raw_data)
        main_handler.invalidate_cache(month)
        cached = main_handler._get_cached(month)
        if not cached:
            continue

        budget_marker = cached.markers.get("budget")
        if not budget_marker:
            continue

        # Scan expense rows (row 2 to budget_marker - 2)
        for row_idx in range(1, budget_marker - 2):
            if row_idx >= len(cached.raw_data):
                break
            row = cached.raw_data[row_idx]
            if len(row) < 6:
                continue

            cat = str(row[5]).strip() if row[5] else ""  # F = category (index 5)
            sub = str(row[3]).strip() if row[3] else ""  # D = subcategory (index 3)

            # Category rename
            if cat in req.renames:
                new_cat = req.renames[cat]
                updates.append((f"F{row_idx + 1}", new_cat))

            # Category removal → replacement
            elif cat in req.removal_replacements:
                updates.append((f"F{row_idx + 1}", req.removal_replacements[cat]))

            # Subcategory rename (within a category)
            effective_cat = req.renames.get(cat, cat)
            if effective_cat in req.sub_renames:
                sub_map = req.sub_renames[effective_cat]
                if sub in sub_map:
                    updates.append((f"D{row_idx + 1}", sub_map[sub]))

        # Scan budget section for category/subcategory labels
        income_marker = cached.markers.get("income", budget_marker + 30)
        for row_idx in range(budget_marker - 1, min(income_marker - 1, len(cached.raw_data))):
            row = cached.raw_data[row_idx]
            if len(row) < 7:
                continue

            cat = str(row[6]).strip() if row[6] else ""  # G = category (index 6)
            sub = str(row[5]).strip() if row[5] else ""  # F = subcategory (index 5)

            if cat in req.renames:
                updates.append((f"G{row_idx + 1}", req.renames[cat]))
            elif cat in req.removal_replacements:
                updates.append((f"G{row_idx + 1}", req.removal_replacements[cat]))

            effective_cat = req.renames.get(cat, cat)
            if effective_cat in req.sub_renames:
                sub_map = req.sub_renames[effective_cat]
                if sub in sub_map:
                    updates.append((f"F{row_idx + 1}", sub_map[sub]))

        if updates:
            main_handler.batch_update_cells(month, updates)
            total_updates += len(updates)
            logger.info(f"Propagated {len(updates)} category updates to {month}")

    # Update lookup tables
    lookup_updates = 0
    if req.renames or req.sub_renames or req.removal_replacements:
        from depensage.classifier.cc_lookup import LookupClassifier
        from depensage.classifier.bank_lookup import BankLookupClassifier
        from depensage.classifier.income_lookup import IncomeLookupClassifier

        for cls in [LookupClassifier(), BankLookupClassifier(), IncomeLookupClassifier()]:
            changed = False
            for name, classification in list(cls.exact.items()):
                cat = classification.category
                sub = getattr(classification, 'subcategory', '') or getattr(classification, 'comments', '')

                new_cat = req.renames.get(cat, req.removal_replacements.get(cat))
                if new_cat:
                    classification.category = new_cat
                    changed = True
                    lookup_updates += 1

                effective_cat = new_cat or cat
                if effective_cat in req.sub_renames:
                    new_sub = req.sub_renames[effective_cat].get(sub)
                    if new_sub:
                        if hasattr(classification, 'subcategory'):
                            classification.subcategory = new_sub
                        elif hasattr(classification, 'comments'):
                            classification.comments = new_sub
                        changed = True
                        lookup_updates += 1

            if changed:
                cls.save()

    return {
        "status": "propagated",
        "cell_updates": total_updates,
        "lookup_updates": lookup_updates,
    }
