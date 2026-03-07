# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DepenSage (dépense + sage) is a Python tool that automates household expense tracking in a Google Sheets spreadsheet. The spreadsheet is in Hebrew and tracks monthly expenses, income, savings goals, and budget reconciliation against actual bank balances. The user (and his wife) download credit card statements, and DepenSage classifies transactions, writes them to the correct monthly sheet, and helps maintain budget accuracy.

## Development Setup

```bash
# Virtual environment (managed with uv, Python 3.12)
source .venv/bin/activate

# Install for development
uv pip install -e .

# Run all tests
python -m unittest discover

# Run a single test file
python -m unittest depensage/config/tests/test_settings.py

# Run a specific test
python -m unittest depensage.config.tests.test_settings.TestLoadSettings.test_load_settings_success

# Lint and format
flake8 depensage/
black depensage/
```

## CLI

The main interface for DepenSage:

```bash
# Process CC statements (the main pipeline)
python -m depensage.sheets.cli process statement1.xlsx [statement2.xlsx ...]
python -m depensage.sheets.cli --year 2026 process statement.xlsx  # only 2026 transactions

# Review unknown merchants interactively
python -m depensage.sheets.cli review statement.xlsx

# Build lookup table from historical sheet data
python -m depensage.sheets.cli build-lookup [--output PATH]

# Consolidate exact entries into prefix patterns
python -m depensage.sheets.cli consolidate-patterns

# Sheet inspection (development)
python -m depensage.sheets.cli list-sheets
python -m depensage.sheets.cli read <sheet> <range>
python -m depensage.sheets.cli formulas <sheet> <range>
python -m depensage.sheets.cli metadata
```

Override defaults with `--spreadsheet-id` and `--credentials` flags. Credentials (service account JSON key) live in `.secrets/` (gitignored).

**Use subagents when reading sheet data** to avoid bloating the main context window with raw spreadsheet content. The subagent runs CLI commands, processes the data, and returns only a concise summary.

## Spreadsheet Structure

The spreadsheet is entirely in Hebrew. Key concepts:

### Monthly Sheets (January–December)
Each month has ~130 expense rows with columns (right-to-left):
- **G:** תאריך (Date) — stored as serial numbers
- **F:** קטגוריה (Category) — from dropdown linked to Categories sheet
- **E:** כמה (Amount) — in NIS
- **D:** תת קטגוריה (Sub-category)
- **C:** הערות (Notes)
- **B:** שם בית עסק (Business name) — populated since March 2025

Below the transactions, each monthly sheet has four sections:
1. **Budget summary** (~row 130) — SUMIFS comparing actual spend per category against budget, with accumulated carryover from prior months
2. **Income** (~row 158) — salary and other income entries
3. **Savings tracker** (~row 167) — per-goal balances with targets and months-remaining
4. **Bank reconciliation** (bottom) — actual bank balances vs. expected. This is where the budget is verified against real money in the bank. The "gap" (פער) reveals tracking errors.

### Green Color-Coding (Charged vs. Pending)
Expenses highlighted green have been charged to the bank account. Non-green expenses are still pending on the credit card. This distinction is structurally important — the reconciliation section only works correctly when charged expenses are marked. A custom **Apps Script `sumbycolor` function** sums green-highlighted cells. This function must be manually triggered by editing its cell.

### Meta Sheets
- **Categories** — master list of 14 expense categories with sub-categories (reference data for dropdowns)
- **Budget** — monthly budget per category/sub-category
- **Month Template** — skeleton duplicated to create new monthly sheets
- **Merged Expenses** — ARRAYFORMULA concatenating all monthly expense data (must be manually updated at year-end to include new months)
- **Expenses so far / Income so far** — annual pivot summaries with monthly averages
- **Budget Planning** — compares prior year plan vs. actuals, sets next year's budgets, models savings and compound interest projections
- **User guide** — Hebrew operational manual for maintaining the spreadsheet
- **`_B` sheets** — wife's former business (closed late 2025, not needed going forward)

### Category List (Hebrew → English)
חשבונות (Bills), בריאות (Health), אלוני (Aloni — child), צ'ופי (Chuppy — dog), עסק (Business — closed), שונות (Miscellaneous — has many sub-categories with individual budgets), סופר (Supermarket), נסיעות (Transportation), בילויים וביזבוזים (Entertainment), משכנתא (Mortgage), שכר דירה (Rent), חסכון (Savings), טיפול (Therapy), יוגה (Yoga)

## Architecture

### 3-Layer Design
Core library (no I/O, no prompts) → CLI (dev/testing) → web app (end product). All business logic lives in core modules. CLI and future web app are thin wrappers.

### Classification: Lookup Table + Human Review
1. **Exact/prefix lookup table** built from historical sheet data (merchant name → category). Handles ~80-90% of transactions.
2. **Human review** for unknown merchants via CLI (`review` command), feeding back into the lookup table.
3. Unclassified transactions are written to the sheet with empty categories — they can be categorized later.

### Automated Pipeline
`engine/pipeline.py` → `run_pipeline()` orchestrates the full flow:
1. Parse Excel CC statements (`StatementParser`)
2. Filter pending transactions (no charge date = still on credit card)
3. Classify via `LookupClassifier` (exact + prefix pattern matching)
4. Deduplicate against existing sheet data (`engine/dedup.py`)
5. Format rows for columns B–G (`engine/formatter.py`)
6. Insert rows if expense section is full, write to correct monthly sheet

The pipeline supports multiple spreadsheets (one per year) and an optional `--year` filter. Without `--year`, transactions are routed to the appropriate year's spreadsheet automatically.

Expense section boundary is marked by `---END---` in column B (at the totals row). Migration script: `scripts/plant_markers.py`.

### Modules
- **`engine/`** — `StatementParser` (Excel only), `pipeline.py` (orchestrator), `dedup.py`, `formatter.py`
- **`classifier/`** — `LookupClassifier` with exact matches and prefix patterns, persisted to `.artifacts/lookup.json`
- **`sheets/`** — `SheetHandler` (Google Sheets API: auth, read, write, metadata, row insertion, marker detection). `cli.py` is the CLI.
- **`config/`** — Settings loaded from `.secrets/config.json`. Config maps years to spreadsheet IDs: `{"spreadsheets": {"2025": "id1", "2026": "id2"}, "credentials_file": "..."}`
- **`scripts/`** — One-time migration scripts

### Key Pain Points to Automate
1. **Data entry** — parsing CC statements and writing to the correct monthly sheet with classified categories
2. **Charged vs. pending tracking** — replacing fragile color-based system with a reliable status mechanism
3. **Month-to-month carryover** — savings balances and accumulated budget are currently copied by hand
4. **Bank reconciliation** — direct bank charges (הוראות קבע) can be auto-tagged since they're recurring and predictable
5. **Merged Expenses formula** — needs manual update each year to include new months

## Future Improvements

- **Wife's CC upload interface** — currently both CC reports must go through the user, which is a bottleneck. A simple web interface (or similar) where the wife can drop her CC report from her phone, and DepenSage picks it up automatically.
- **Automated CC download** — Israeli CC providers (Cal, Max, Isracard) may have APIs or scrapeable portals that could eliminate manual download entirely.
- **Recurring charge templates** — direct bank charges (mortgage, insurance, ארנונה, etc.) repeat monthly and could be pre-populated automatically.
- **Budget alerts** — mid-month notifications when a category is approaching its budget limit.
- **Automatic month setup** — creating the new month sheet, carrying over savings/accumulated balances, and updating the Merged Expenses formula are all currently manual.

## Key Conventions

- Tests use `unittest`, located in `<module>/tests/` subdirectories
- External dependencies (Google Sheets API) are mocked in tests
- Transactions flow as pandas DataFrames with columns: `date`, `business_name`, `amount`, `category`, `subcategory`
- Sheet dates are serial numbers (days since 1899-12-30) when read with `UNFORMATTED_VALUE`. Dedup and `read_expense_rows` handle this.
- The spreadsheet is in Hebrew; all category/sub-category names are Hebrew strings
- Configuration and secrets are stored in `.secrets/` (gitignored): `config.json` for settings, service account JSON for credentials
- Generated artifacts (lookup tables, etc.) go in `.artifacts/` (gitignored)
- Never hardcode spreadsheet IDs, credentials paths, or anything personally identifiable in committed code — keep it in `.secrets/`, `.artifacts/`, or environment variables
- Commit early and often — small, focused commits that each do one thing
- Commit messages should be succinct (one short line)
