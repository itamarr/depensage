"""
Pydantic models for API request/response serialization.
"""

from pydantic import BaseModel


class LoginRequest(BaseModel):
    password: str


class LoginResponse(BaseModel):
    ok: bool
    message: str = ""


class UploadResponse(BaseModel):
    session_id: str
    files: list[str]


class PipelineRunRequest(BaseModel):
    spreadsheet_key: str


class ProgressEvent(BaseModel):
    stage: str
    message: str
    percent: int


class MonthStageSummary(BaseModel):
    month: str
    year: int
    is_new: bool
    new_expenses: int
    duplicates: int
    new_income: int
    income_duplicates: int
    bank_balance: float | None
    savings_allocations: int
    savings_warning: str | None
    carryover_updates: int


class StagedResultSummary(BaseModel):
    total_parsed: int
    in_process_skipped: int
    classified: int
    unclassified: int
    unclassified_merchants: list[str]
    months: list[MonthStageSummary]
    has_writes: bool


class ExpenseRow(BaseModel):
    index: int
    business_name: str
    notes: str
    subcategory: str
    amount: str
    category: str
    date: str
    status: str


class IncomeRow(BaseModel):
    index: int
    comments: str
    amount: str
    category: str
    date: str


class SavingsAllocationItem(BaseModel):
    goal_name: str
    allocated: float
    preset_incoming: float
    target: float
    current_total: float
    is_default: bool
    is_blatam: bool


class MonthStageDetail(BaseModel):
    month: str
    year: int
    is_new: bool
    expenses: list[ExpenseRow]
    income: list[IncomeRow]
    bank_balance: float | None
    savings_allocations: list[SavingsAllocationItem]
    savings_warning: str | None
    carryover_updates: int
    duplicates: int
    income_duplicates: int


class ExpenseEdit(BaseModel):
    index: int
    category: str
    subcategory: str


class IncomeEdit(BaseModel):
    index: int
    category: str
    comments: str = ""


class SavingsEdit(BaseModel):
    goal_name: str
    allocated: float


class BulkEditRequest(BaseModel):
    expenses: list[ExpenseEdit] = []
    income: list[IncomeEdit] = []
    savings: list[SavingsEdit] = []
    bank_balance: float | None = None


class CategoryInfo(BaseModel):
    categories: dict[str, list[str]]


class RowChangeItem(BaseModel):
    month: str
    row_type: str
    source: str
    lookup_key: str
    old_category: str
    new_category: str
    old_subcategory: str
    date: str = ""
    new_subcategory: str


class LookupUpdateRequest(BaseModel):
    changes: list[RowChangeItem]


class CommitResult(BaseModel):
    total_parsed: int
    months: list[dict]


class HealthResponse(BaseModel):
    status: str
    version: str


class ConfigResponse(BaseModel):
    spreadsheets: dict[str, dict]
    years: list[int]
