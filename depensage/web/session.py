"""
In-memory pipeline session store with TTL.

Each pipeline run gets a session that holds the staged result,
handlers, temp files, and classification state. Sessions expire
after 2 hours and are cleaned up by a background task.
"""

import shutil
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from depensage.engine.staging import StagedPipelineResult, RowChange
from depensage.sheets.spreadsheet_handler import SheetHandler


@dataclass
class PipelineSession:
    session_id: str
    created_at: datetime
    temp_dir: str
    uploaded_files: list[str] = field(default_factory=list)
    staged_result: StagedPipelineResult | None = None
    handlers: dict[int, SheetHandler] | None = None
    year_filter: int | None = None
    categories: dict | None = None
    changes: list[RowChange] | None = None
    status: str = "created"  # created | uploaded | running | staged | committed
    error: str | None = None
    progress_stage: str = ""
    progress_pct: int = 0


class SessionStore:
    """Thread-safe in-memory session store with TTL."""

    def __init__(self, ttl_hours: int = 2):
        self._sessions: dict[str, PipelineSession] = {}
        self._lock = threading.Lock()
        self._ttl = timedelta(hours=ttl_hours)

    def create(self, temp_dir: str) -> PipelineSession:
        session_id = uuid.uuid4().hex[:12]
        session = PipelineSession(
            session_id=session_id,
            created_at=datetime.now(),
            temp_dir=temp_dir,
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> PipelineSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def delete(self, session_id: str):
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session:
            _cleanup_session(session)

    def sweep_expired(self):
        now = datetime.now()
        expired = []
        with self._lock:
            for sid, session in list(self._sessions.items()):
                if now - session.created_at > self._ttl:
                    expired.append(self._sessions.pop(sid))
        for session in expired:
            _cleanup_session(session)


def _cleanup_session(session: PipelineSession):
    """Remove temp files for an expired or discarded session."""
    try:
        shutil.rmtree(session.temp_dir, ignore_errors=True)
    except Exception:
        pass
