# Pipeline & Engine Details

## Automated Pipeline

`engine/pipeline.py` → `run_pipeline()` orchestrates the full flow using in-memory `VirtualMonth` objects. **Zero writes to Google Sheets during staging** — the API is used only for reads. All writes happen at commit time.

### Sequential Per-Month Processing

Months are processed in chronological order, each fully finalized before the next starts:

1. **Parse** all input files (CC statements + bank transcripts, auto-detected)
2. **Classify** transactions (CC, bank expenses, income)
3. **For each month** (chronological):
   a. **Load VirtualMonth** — from existing sheet (read-only) or template (for new months)
   b. **Carryover** — if accumulated columns are empty, compute from previous month's VM
   c. **Dedup + stage expenses** — against VM's existing data
   d. **Dedup + stage income** — against VM's existing data
   e. **Update VM** — add staged expenses/income so next month sees accurate data
   f. **Stage bank balance** — from bank transcript
   g. **Stage savings allocation** — using post-carryover savings budget
4. **Return StagedPipelineResult** — caller can inspect, export XLSX, and commit

### VirtualMonth (`engine/virtual_month.py`)

In-memory representation of a month sheet, with Python equivalents of spreadsheet formulas:

- `load_from_sheet(handler, month, year)` — reads existing sheet data
- `load_from_template(handler, month, year)` — reads template structure, marks `is_new=True`
- `BudgetLine` / `SavingsLine` dataclasses hold section data
- `compute_budget_remaining()`, `compute_savings_total()`, `compute_income_total()` — replace spreadsheet SUMIF/arithmetic

### Commit Flow (`staging.py`)

For each month (chronological):
1. **Create sheet** from template (if `is_new`)
2. **Write carryover** formulas/values (budget accumulated, savings accumulated, savings budget)
3. **Insert rows** if expense/income sections are full
4. **Write expenses** (B:H)
5. **Write income** (D:G)
6. **Batch write** savings allocations + bank balance

## Classification: Lookup Table + Human Review

1. **Exact/prefix lookup table** built from historical sheet data (merchant name → category). Handles ~80-90% of transactions.
2. **Human review** for unknown merchants via CLI (`review` command), feeding back into the lookup table.
3. Unclassified transactions are written to the sheet with empty categories — they can be categorized later.

## Month-to-Month Carryover

`engine/carryover.py` provides two entry points:
- `compute_carryover(source_vm, dest_vm, same_spreadsheet)` — pure computation from VirtualMonth objects (used by pipeline)
- `run_carryover(source_handler, source_month, dest_handler, dest_month)` — direct read-and-write (used by manual `carryover` CLI command)

### What carries over (source → destination):
- **Budget accumulated**: For CARRY-flagged lines only, positive remaining (surplus) from source → Accumulated (E) column in destination. Debts do NOT carry — they're handled by the surplus/deficit formula.
- **Savings accumulated**: Total (C) from each savings goal in source → Accumulated (F) in destination. Matched by goal name (cross-year) or row number (same-spreadsheet).
- **Savings budget line**: The חסכון budget line in destination is set so that total budget = source month's income total. Formula: `savings_budget = income_total - sum(all other budget D values)`.

### Same-spreadsheet vs. cross-year:
- **Same spreadsheet** (e.g., Jan → Feb): writes formulas (`=MAX('January'!B134,0)`) so values update live
- **Cross-year** (e.g., Dec 2025 → Jan 2026): writes static values matched by (category, subcategory) for budget, by goal name for savings

### Carryover detection:
The pipeline runs carryover for any month where accumulated columns are all empty — whether the sheet is new or was created manually. This is checked by `_needs_carryover(vm)`.

## Savings Allocation

`engine/savings_allocator.py` → `allocate_savings()` is a pure function:
- **Good month** (budget ≥ presets): keep presets, adjust בלת"ם, surplus → default goal
- **Tight month** (0 < budget < presets): keep presets, warn user
- **Bad month** (budget ≤ 0): zero all allocations, warn user

The pipeline reads savings budget and goals from the VirtualMonth (post-carryover), so allocations reflect the correct savings budget.

## Deduplication

`engine/dedup.py` compares incoming transactions against existing sheet rows using `(date, business_name, amount)` tuples. Sheet dates are serial numbers when read with `UNFORMATTED_VALUE` — the dedup module handles conversion.

## Statement Parsing

`engine/statement_parser.py` parses Israeli CC Excel statements (.xlsx). Detects columns by header text. Extracts: date, business_name, amount, charge_date. The `filter_pending()` static method removes transactions without a charge date.

## Modules

- **`engine/`** — `virtual_month.py` (in-memory month), `pipeline.py` (sequential orchestrator), `carryover.py` (compute + apply), `savings_allocator.py`, `staging.py` (MonthStage + XLSX export/import), `dedup.py`, `formatter.py`, `statement_parser.py`, `bank_parser.py`, `verification.py`
- **`classifier/`** — `LookupClassifier` with exact matches and prefix patterns, persisted to `.artifacts/lookup.json`. Bank and income classifiers with same pattern.
- **`sheets/`** — `SheetHandler` (Google Sheets API: auth, read, write, metadata, row insertion, marker detection). `cli.py` + `cli_commands.py` for the CLI.
- **`config/`** — Settings loaded from `.secrets/config.json`. Config maps keys to spreadsheet IDs: `{"spreadsheets": {"2025": {...}, "2026_dev": {...}}, "credentials_file": "...", "default_savings_goal": "דירה"}`
- **`scripts/`** — One-time migration scripts
