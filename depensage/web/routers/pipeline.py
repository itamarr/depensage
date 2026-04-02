"""
Pipeline endpoints: upload, run, progress SSE, result, commit.
"""

import asyncio
import json
import logging
import os
import tempfile

from fastapi import APIRouter, Request, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

from depensage.config.settings import load_settings, get_spreadsheet_entry, get_entries_for_year
from depensage.sheets.spreadsheet_handler import SheetHandler
from depensage.classifier.cc_lookup import LookupClassifier
from depensage.classifier.bank_lookup import BankLookupClassifier
from depensage.classifier.income_lookup import IncomeLookupClassifier
from depensage.engine.pipeline import run_pipeline
from depensage.web.auth import require_auth
from depensage.web.models import (
    UploadResponse, PipelineRunRequest, StagedResultSummary,
    MonthStageSummary, CommitResult, MonthStageDetail,
    ExpenseRow, IncomeRow, SavingsAllocationItem,
)
from depensage.web.session import SessionStore

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/pipeline", tags=["pipeline"],
    dependencies=[Depends(require_auth)],
)


def _get_store(request: Request) -> SessionStore:
    return request.app.state.session_store


@router.get("/history")
async def get_history(request: Request):
    """Get recent pipeline run history."""
    history_path = os.path.join(".artifacts", "run_history.json")
    if not os.path.exists(history_path):
        return {"runs": []}
    try:
        with open(history_path) as f:
            runs = json.load(f)
        return {"runs": runs[-20:]}  # Last 20
    except Exception:
        return {"runs": []}


@router.post("/upload", response_model=UploadResponse)
async def upload_files(
    request: Request,
    files: list[UploadFile] = File(...),
):
    """Upload statement files and create a pipeline session."""
    store = _get_store(request)
    temp_dir = tempfile.mkdtemp(prefix="depensage_")

    saved = []
    for f in files:
        if not f.filename or not f.filename.endswith(".xlsx"):
            continue
        dest = os.path.join(temp_dir, f.filename)
        content = await f.read()
        with open(dest, "wb") as out:
            out.write(content)
        saved.append(dest)

    if not saved:
        os.rmdir(temp_dir)
        raise HTTPException(status_code=400, detail="No .xlsx files uploaded")

    session = store.create(temp_dir)
    session.uploaded_files = saved
    session.status = "uploaded"

    return UploadResponse(
        session_id=session.session_id,
        files=[os.path.basename(f) for f in saved],
    )


@router.post("/{session_id}/run")
async def run(
    session_id: str,
    req: PipelineRunRequest,
    request: Request,
):
    """Start pipeline execution. Returns immediately; poll /progress for SSE."""
    store = _get_store(request)
    session = store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status not in ("uploaded", "staged"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot run pipeline in status '{session.status}'",
        )

    # Set up handlers
    settings = load_settings()
    credentials = os.path.abspath(settings["credentials_file"])
    entry = get_spreadsheet_entry(req.spreadsheet_key, settings)
    year = entry.get("year")
    if not year:
        raise HTTPException(
            status_code=400,
            detail=f"Spreadsheet '{req.spreadsheet_key}' has no year",
        )

    handler = SheetHandler(entry["id"])
    handler.authenticate(credentials)
    handlers = {year: handler}

    # Load previous year handler for carryover
    prev_year = year - 1
    prev_entries = get_entries_for_year(prev_year, settings)
    if len(prev_entries) == 1:
        _, prev_entry = prev_entries[0]
        prev_handler = SheetHandler(prev_entry["id"])
        prev_handler.authenticate(credentials)
        handlers[prev_year] = prev_handler

    session.handlers = handlers
    session.year_filter = year
    session.status = "running"
    session.progress_stage = "starting"
    session.progress_pct = 0

    # Fetch categories for the UI
    from depensage.sheets.cli_helpers import fetch_categories
    try:
        session.categories = fetch_categories(handler)
    except Exception:
        session.categories = {}

    # Run pipeline in background thread
    def _run():
        try:
            session.progress_stage = "parsing"
            session.progress_pct = 10

            classifier = LookupClassifier()
            bank_classifier = BankLookupClassifier()
            income_classifier = IncomeLookupClassifier()

            session.progress_stage = "classifying"
            session.progress_pct = 30

            result = run_pipeline(
                session.uploaded_files,
                handlers,
                classifier,
                year=year,
                bank_classifier=bank_classifier,
                income_classifier=income_classifier,
            )

            session.progress_stage = "complete"
            session.progress_pct = 100
            session.staged_result = result
            session.status = "staged"
        except Exception as e:
            logger.exception("Pipeline failed")
            session.status = "staged"  # Allow retry
            session.error = str(e)
            session.progress_stage = "error"
            session.progress_pct = 0

    asyncio.get_event_loop().run_in_executor(None, _run)

    return {"status": "running", "session_id": session_id}


@router.get("/{session_id}/progress")
async def progress(session_id: str, request: Request):
    """SSE stream for pipeline progress."""
    store = _get_store(request)
    session = store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    async def event_stream():
        last_stage = ""
        while True:
            if session.progress_stage != last_stage:
                last_stage = session.progress_stage
                data = json.dumps({
                    "stage": session.progress_stage,
                    "percent": session.progress_pct,
                    "error": session.error,
                })
                yield f"data: {data}\n\n"
                if session.progress_stage in ("complete", "error"):
                    break
            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{session_id}/result", response_model=StagedResultSummary)
async def result(
    session_id: str,
    request: Request,
):
    """Get staged pipeline result summary."""
    store = _get_store(request)
    session = store.get(session_id)
    if not session or not session.staged_result:
        raise HTTPException(status_code=404, detail="No staged result")

    staged = session.staged_result
    months = []
    for stage in staged.sorted_stages():
        months.append(MonthStageSummary(
            month=stage.month,
            year=stage.year,
            is_new=stage.is_new,
            new_expenses=len(stage.new_expenses),
            duplicates=stage.duplicates,
            new_income=len(stage.new_income),
            income_duplicates=stage.income_duplicates,
            bank_balance=stage.bank_balance,
            savings_allocations=len(stage.savings_allocations),
            savings_warning=stage.savings_warning,
            carryover_updates=len(stage.carryover_updates),
        ))

    return StagedResultSummary(
        total_parsed=staged.total_parsed,
        in_process_skipped=staged.in_process_skipped,
        classified=staged.classified,
        unclassified=staged.unclassified,
        unclassified_merchants=staged.unclassified_merchants,
        months=months,
        has_writes=staged.has_writes(),
    )


@router.get("/{session_id}/months/{month}/{year}", response_model=MonthStageDetail)
async def month_detail(
    session_id: str,
    month: str,
    year: int,
    request: Request,
):
    """Get detailed staged data for a specific month."""
    store = _get_store(request)
    session = store.get(session_id)
    if not session or not session.staged_result:
        raise HTTPException(status_code=404, detail="No staged result")

    key = (month, year)
    stage = session.staged_result.month_stages.get(key)
    if not stage:
        raise HTTPException(status_code=404, detail=f"No stage for {month} {year}")

    expenses = []
    for i, row in enumerate(stage.new_expenses):
        expenses.append(ExpenseRow(
            index=i,
            business_name=str(row[0]) if row[0] else "",
            notes=str(row[1]) if len(row) > 1 and row[1] else "",
            subcategory=str(row[2]) if len(row) > 2 and row[2] else "",
            amount=str(row[3]) if len(row) > 3 and row[3] else "",
            category=str(row[4]) if len(row) > 4 and row[4] else "",
            date=str(row[5]) if len(row) > 5 and row[5] else "",
            status=str(row[6]) if len(row) > 6 and row[6] else "",
        ))

    income = []
    for i, row in enumerate(stage.new_income):
        income.append(IncomeRow(
            index=i,
            comments=str(row[0]) if row[0] else "",
            amount=str(row[1]) if len(row) > 1 and row[1] else "",
            category=str(row[2]) if len(row) > 2 and row[2] else "",
            date=str(row[3]) if len(row) > 3 and row[3] else "",
        ))

    allocs = []
    for a in stage.savings_allocations:
        allocs.append(SavingsAllocationItem(
            goal_name=a.goal_name,
            allocated=a.allocated,
            preset_incoming=a.preset_incoming,
            target=a.target,
            current_total=a.current_total,
            is_default=a.is_default,
            is_blatam=a.is_blatam,
        ))

    return MonthStageDetail(
        month=stage.month,
        year=stage.year,
        is_new=stage.is_new,
        expenses=expenses,
        income=income,
        bank_balance=stage.bank_balance,
        savings_allocations=allocs,
        savings_warning=stage.savings_warning,
        carryover_updates=len(stage.carryover_updates),
        duplicates=stage.duplicates,
        income_duplicates=stage.income_duplicates,
    )


@router.post("/{session_id}/commit", response_model=CommitResult)
async def commit(
    session_id: str,
    request: Request,
):
    """Commit staged data to Google Sheets."""
    store = _get_store(request)
    session = store.get(session_id)
    if not session or not session.staged_result:
        raise HTTPException(status_code=404, detail="No staged result")
    if not session.handlers:
        raise HTTPException(status_code=400, detail="No handlers configured")

    try:
        result = session.staged_result.commit(session.handlers)
    except Exception as e:
        logger.exception("Commit failed")
        raise HTTPException(status_code=500, detail=str(e))

    session.status = "committed"

    # Persist to run history
    _save_run_history(session, result)

    months = []
    for mr in result.months:
        months.append({
            "month": mr.month,
            "year": mr.year,
            "written": mr.written,
            "duplicates": mr.duplicates,
            "income_written": mr.income_written,
            "income_duplicates": mr.income_duplicates,
        })

    return CommitResult(total_parsed=result.total_parsed, months=months)


@router.delete("/{session_id}")
async def discard(
    session_id: str,
    request: Request,
):
    """Discard a pipeline session and clean up temp files."""
    store = _get_store(request)
    store.delete(session_id)
    return {"status": "deleted"}


def _save_run_history(session, result):
    """Append to .artifacts/run_history.json."""
    import json
    from datetime import datetime

    history_path = os.path.join(".artifacts", "run_history.json")
    os.makedirs(".artifacts", exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "files": [os.path.basename(f) for f in session.uploaded_files],
        "spreadsheet_year": session.year_filter,
        "total_parsed": result.total_parsed,
        "months": [
            {"month": mr.month, "year": mr.year,
             "written": mr.written, "income_written": mr.income_written}
            for mr in result.months
        ],
        "status": "committed",
    }

    history = []
    if os.path.exists(history_path):
        try:
            with open(history_path) as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append(entry)
    # Keep last 50 entries
    history = history[-50:]

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
