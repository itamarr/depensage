"""
Staging endpoints: edit staged data, classify unknowns, detect changes,
apply lookup updates.
"""

import logging

from fastapi import APIRouter, Request, Depends, HTTPException

from depensage.engine.staging import RowMeta, RowChange
from depensage.web.auth import require_auth
from depensage.web.models import (
    BulkEditRequest, CategoryInfo, RowChangeItem, LookupUpdateRequest,
    ExpenseRow, IncomeRow,
)
from depensage.web.session import SessionStore

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/staging", tags=["staging"],
    dependencies=[Depends(require_auth)],
)


def _get_store(request: Request) -> SessionStore:
    return request.app.state.session_store


def _get_session(request, session_id):
    store = _get_store(request)
    session = store.get(session_id)
    if not session or not session.staged_result:
        raise HTTPException(status_code=404, detail="No staged result")
    return session


@router.get("/{session_id}/categories", response_model=CategoryInfo)
async def get_categories(session_id: str, request: Request):
    """Get category → subcategory mapping for the classification UI."""
    session = _get_session(request, session_id)
    return CategoryInfo(categories=session.categories or {})


@router.get("/{session_id}/unknowns")
async def get_unknowns(session_id: str, request: Request):
    """Get unclassified expense rows across all months."""
    session = _get_session(request, session_id)
    staged = session.staged_result

    unknowns = []
    for stage in staged.sorted_stages():
        for i, row in enumerate(stage.new_expenses):
            cat = str(row[4]) if len(row) > 4 and row[4] else ""
            if not cat:
                unknowns.append({
                    "month": stage.month,
                    "year": stage.year,
                    "index": i,
                    "business_name": str(row[0]) if row[0] else "",
                    "amount": str(row[3]) if len(row) > 3 and row[3] else "",
                    "date": str(row[5]) if len(row) > 5 and row[5] else "",
                    "status": str(row[6]) if len(row) > 6 and row[6] else "",
                })

    # Find prefix groups for suggestions
    from depensage.sheets.cli_helpers import find_prefix_groups
    merchant_names = [u["business_name"] for u in unknowns if u["business_name"]]
    groups = find_prefix_groups(merchant_names) if len(merchant_names) >= 2 else []

    return {"unknowns": unknowns, "prefix_groups": groups}


@router.put("/{session_id}/classify")
async def classify_unknowns(
    session_id: str,
    request: Request,
    edits: BulkEditRequest,
):
    """Bulk-classify expenses by setting category/subcategory."""
    session = _get_session(request, session_id)
    staged = session.staged_result
    updated = 0

    for edit in edits.expenses:
        # Find the right stage by scanning all months
        for stage in staged.sorted_stages():
            if edit.index < len(stage.new_expenses):
                row = stage.new_expenses[edit.index]
                if len(row) > 4:
                    row[4] = edit.category
                if len(row) > 2:
                    row[2] = edit.subcategory
                # Update meta
                if edit.index < len(stage.expense_meta):
                    stage.expense_meta[edit.index].needs_review = not edit.category
                updated += 1
                break

    return {"updated": updated}


@router.put("/{session_id}/months/{month}/{year}")
async def edit_month(
    session_id: str,
    month: str,
    year: int,
    request: Request,
    edits: BulkEditRequest,
):
    """Edit staged expenses and income for a specific month."""
    session = _get_session(request, session_id)
    staged = session.staged_result

    key = (month, year)
    stage = staged.month_stages.get(key)
    if not stage:
        raise HTTPException(status_code=404, detail=f"No stage for {month} {year}")

    updated = 0

    for edit in edits.expenses:
        if 0 <= edit.index < len(stage.new_expenses):
            row = stage.new_expenses[edit.index]
            if len(row) > 4:
                row[4] = edit.category
            if len(row) > 2:
                row[2] = edit.subcategory
            updated += 1

    for edit in edits.income:
        if 0 <= edit.index < len(stage.new_income):
            row = stage.new_income[edit.index]
            if len(row) > 2:
                row[2] = edit.category
            if len(row) > 0:
                row[0] = edit.comments
            updated += 1

    for edit in edits.savings:
        for alloc in stage.savings_allocations:
            if alloc.goal_name == edit.goal_name:
                alloc.allocated = edit.allocated
                updated += 1
                break

    if edits.bank_balance is not None:
        stage.bank_balance = edits.bank_balance
        updated += 1

    return {"updated": updated}


@router.get("/{session_id}/changes")
async def get_changes(session_id: str, request: Request):
    """Detect lookup changes by comparing current staged data against original metadata."""
    session = _get_session(request, session_id)
    staged = session.staged_result
    changes = []

    for stage in staged.sorted_stages():
        # Check expenses
        for i, row in enumerate(stage.new_expenses):
            if i >= len(stage.expense_meta):
                break
            meta = stage.expense_meta[i]
            new_cat = str(row[4]) if len(row) > 4 and row[4] else ""
            new_sub = str(row[2]) if len(row) > 2 and row[2] else ""

            if new_cat != meta.orig_category or new_sub != meta.orig_subcategory:
                status = str(row[6]) if len(row) > 6 and row[6] else ""
                source = "bank" if status == "BANK" else "cc"
                changes.append(RowChangeItem(
                    month=stage.month,
                    row_type="expense",
                    source=source,
                    lookup_key=str(row[0]) if row[0] else "",
                    old_category=meta.orig_category,
                    new_category=new_cat,
                    old_subcategory=meta.orig_subcategory,
                    new_subcategory=new_sub,
                    date=str(row[5]) if len(row) > 5 and row[5] else "",
                ))

        # Check income
        for i, row in enumerate(stage.new_income):
            if i >= len(stage.income_meta):
                break
            meta = stage.income_meta[i]
            new_cat = str(row[2]) if len(row) > 2 and row[2] else ""
            new_sub = str(row[0]) if row[0] else ""

            if new_cat != meta.orig_category or new_sub != meta.orig_subcategory:
                changes.append(RowChangeItem(
                    month=stage.month,
                    row_type="income",
                    source="income",
                    lookup_key=str(row[0]) if row[0] else "",
                    old_category=meta.orig_category,
                    new_category=new_cat,
                    old_subcategory=meta.orig_subcategory,
                    new_subcategory=new_sub,
                    date=str(row[3]) if len(row) > 3 and row[3] else "",
                ))

    return {"changes": changes}


@router.put("/{session_id}/lookup-updates")
async def apply_lookup_updates(
    session_id: str,
    request: Request,
    req: LookupUpdateRequest,
):
    """Apply confirmed lookup changes to the classifier files."""
    from depensage.engine.lookup_updater import apply_lookup_updates as do_updates
    from depensage.classifier.cc_lookup import LookupClassifier
    from depensage.classifier.bank_lookup import BankLookupClassifier
    from depensage.classifier.income_lookup import IncomeLookupClassifier

    _ = _get_session(request, session_id)  # auth check

    # Convert to RowChange objects
    row_changes = [
        RowChange(
            month=c.month, row_type=c.row_type, source=c.source,
            lookup_key=c.lookup_key,
            old_category=c.old_category, new_category=c.new_category,
            old_subcategory=c.old_subcategory, new_subcategory=c.new_subcategory,
        )
        for c in req.changes
    ]

    cc = LookupClassifier()
    bank = BankLookupClassifier()
    income = IncomeLookupClassifier()
    updated = do_updates(row_changes, cc, bank, income)

    return {"updated_classifiers": updated}
