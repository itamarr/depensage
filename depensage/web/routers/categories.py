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


def _scan_changes(handler, req: PropagateRequest):
    """Scan all months and return a preview of changes without applying.

    Returns: {month: [(cell_ref, old_value, new_value, section)]}
    and lookup_changes: [(classifier, key, old_cat, new_cat)]
    """
    settings = load_settings()
    years = get_all_years(settings)
    current_year = max(years)

    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]

    month_changes = {}  # month → list of change dicts

    for month in month_names:
        if not handler.sheet_exists(month):
            continue

        handler.invalidate_cache(month)
        cached = handler._get_cached(month)
        if not cached:
            continue

        budget_marker = cached.markers.get("budget")
        if not budget_marker:
            continue

        changes = []

        # Scan expense rows
        for row_idx in range(1, budget_marker - 2):
            if row_idx >= len(cached.raw_data):
                break
            row = cached.raw_data[row_idx]
            if len(row) < 6:
                continue

            cat = str(row[5]).strip() if row[5] else ""
            sub = str(row[3]).strip() if row[3] else ""
            biz = str(row[1]).strip() if len(row) > 1 and row[1] else ""

            if cat in req.renames:
                changes.append({
                    "section": "expense", "cell": f"F{row_idx + 1}",
                    "field": "category", "old": cat, "new": req.renames[cat],
                    "context": biz,
                })
            elif cat in req.removal_replacements:
                changes.append({
                    "section": "expense", "cell": f"F{row_idx + 1}",
                    "field": "category", "old": cat, "new": req.removal_replacements[cat],
                    "context": biz,
                })

            effective_cat = req.renames.get(cat, cat)
            if effective_cat in req.sub_renames:
                new_sub = req.sub_renames[effective_cat].get(sub)
                if new_sub:
                    changes.append({
                        "section": "expense", "cell": f"D{row_idx + 1}",
                        "field": "subcategory", "old": sub, "new": new_sub,
                        "context": biz,
                    })

        # Scan budget section
        income_marker = cached.markers.get("income", budget_marker + 30)
        for row_idx in range(budget_marker - 1, min(income_marker - 1, len(cached.raw_data))):
            row = cached.raw_data[row_idx]
            if len(row) < 7:
                continue

            cat = str(row[6]).strip() if row[6] else ""
            sub = str(row[5]).strip() if row[5] else ""

            if cat in req.renames:
                changes.append({
                    "section": "budget", "cell": f"G{row_idx + 1}",
                    "field": "category", "old": cat, "new": req.renames[cat],
                    "context": "",
                })
            elif cat in req.removal_replacements:
                changes.append({
                    "section": "budget", "cell": f"G{row_idx + 1}",
                    "field": "category", "old": cat, "new": req.removal_replacements[cat],
                    "context": "",
                })

            effective_cat = req.renames.get(cat, cat)
            if effective_cat in req.sub_renames:
                new_sub = req.sub_renames[effective_cat].get(sub)
                if new_sub:
                    changes.append({
                        "section": "budget", "cell": f"F{row_idx + 1}",
                        "field": "subcategory", "old": sub, "new": new_sub,
                        "context": "",
                    })

        if changes:
            month_changes[month] = changes

    # Scan lookups
    lookup_changes = []
    if req.renames or req.sub_renames or req.removal_replacements:
        from depensage.classifier.cc_lookup import LookupClassifier
        from depensage.classifier.bank_lookup import BankLookupClassifier
        from depensage.classifier.income_lookup import IncomeLookupClassifier

        for cls_name, cls in [("cc", LookupClassifier()), ("bank", BankLookupClassifier()), ("income", IncomeLookupClassifier())]:
            for name, classification in cls.exact.items():
                cat = classification.category
                sub = getattr(classification, 'subcategory', '') or getattr(classification, 'comments', '')

                new_cat = req.renames.get(cat, req.removal_replacements.get(cat))
                if new_cat:
                    lookup_changes.append({
                        "classifier": cls_name, "key": name,
                        "field": "category", "old": cat, "new": new_cat,
                    })

                effective_cat = new_cat or cat
                if effective_cat in req.sub_renames:
                    new_sub = req.sub_renames[effective_cat].get(sub)
                    if new_sub:
                        lookup_changes.append({
                            "classifier": cls_name, "key": name,
                            "field": "subcategory", "old": sub, "new": new_sub,
                        })

    return month_changes, lookup_changes


@router.post("/propagate/preview")
async def preview_propagation(req: PropagateRequest):
    """Preview what changes would be made without applying them."""
    main_handler, _ = _get_handlers()
    month_changes, lookup_changes = _scan_changes(main_handler, req)
    total_cells = sum(len(c) for c in month_changes.values())
    return {
        "month_changes": month_changes,
        "lookup_changes": lookup_changes,
        "total_cells": total_cells,
        "total_lookups": len(lookup_changes),
    }


@router.post("/propagate")
async def propagate_renames(req: PropagateRequest):
    """Apply category/subcategory renames across all months and lookups."""
    main_handler, _ = _get_handlers()
    month_changes, lookup_changes = _scan_changes(main_handler, req)

    # Apply month changes
    total_updates = 0
    for month, changes in month_changes.items():
        updates = [(c["cell"], c["new"]) for c in changes]
        if updates:
            main_handler.invalidate_cache(month)
            main_handler.batch_update_cells(month, updates)
            total_updates += len(updates)
            logger.info(f"Propagated {len(updates)} category updates to {month}")

    # Apply lookup changes
    lookup_updates = 0
    if lookup_changes:
        from depensage.classifier.cc_lookup import LookupClassifier
        from depensage.classifier.bank_lookup import BankLookupClassifier
        from depensage.classifier.income_lookup import IncomeLookupClassifier

        classifiers = {"cc": LookupClassifier(), "bank": BankLookupClassifier(), "income": IncomeLookupClassifier()}
        changed_cls = set()

        for lc in lookup_changes:
            cls = classifiers[lc["classifier"]]
            name = lc["key"]
            if name in cls.exact:
                classification = cls.exact[name]
                if lc["field"] == "category":
                    classification.category = lc["new"]
                elif lc["field"] == "subcategory":
                    if hasattr(classification, 'subcategory'):
                        classification.subcategory = lc["new"]
                    elif hasattr(classification, 'comments'):
                        classification.comments = lc["new"]
                changed_cls.add(lc["classifier"])
                lookup_updates += 1

        for cls_name in changed_cls:
            classifiers[cls_name].save()

    return {
        "status": "propagated",
        "cell_updates": total_updates,
        "lookup_updates": lookup_updates,
    }
