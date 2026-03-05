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

## Sheets CLI (Development Tool)

A CLI for inspecting and interacting with the Google Sheet during development:

```bash
python -m depensage.sheets.cli list-sheets
python -m depensage.sheets.cli read <sheet> <range>
python -m depensage.sheets.cli formulas <sheet> <range>
python -m depensage.sheets.cli metadata
```

Defaults are configured in `depensage/sheets/cli.py` (playground spreadsheet ID and credentials path). Override with `--spreadsheet-id` and `--credentials` flags. Credentials (service account JSON key) live in `.secrets/` (gitignored).

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

## Architecture (Current State)

The codebase was scaffolded about a year ago and is being rethought. Key decisions:

### Classification: Lookup Table + LLM (replacing Neural Network)
The original TensorFlow neural classifier (`classifier/` module) is being replaced. The new approach:
1. **Exact/fuzzy lookup table** built from historical sheet data (merchant name → category). Handles ~80-90% of transactions.
2. **LLM fallback** for unknown merchants — a single API call with merchant name, amount, and category list. The LLM integration should be generic (not locked to a specific provider).
3. **Human review** for low-confidence results, feeding back into the lookup table.

TensorFlow and scikit-learn have been removed from dependencies.

### Existing Modules
- **`sheets/`** — Google Sheets API integration via service account. `SheetHandler` handles auth, read, write, metadata, template-based sheet creation. `cli.py` is the development CLI.
- **`engine/`** — `ExpenseProcessor` orchestrates the pipeline; `StatementParser` parses CSV/Excel credit card statements.
- **`config/`** — Settings loaded from `~/.depensage/config.json`. Singleton pattern.
- **`classifier/`** — Legacy neural classifier (to be replaced).

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
- The spreadsheet is in Hebrew; all category/sub-category names are Hebrew strings
- Configuration and secrets are stored in `.secrets/` (gitignored): `config.json` for settings, service account JSON for credentials
- Never hardcode spreadsheet IDs, credentials paths, or anything personally identifiable in committed code — keep it in `.secrets/` or environment variables
- Commit early and often — small, focused commits that each do one thing
- Commit messages should be succinct (one short line)
