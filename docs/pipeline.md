# Pipeline & Engine Details

## Automated Pipeline

`engine/pipeline.py` → `run_pipeline()` orchestrates the full flow:

1. Parse Excel CC statements (`StatementParser`)
2. Filter pending transactions (no charge date = still on credit card)
3. Classify via `LookupClassifier` (exact + prefix pattern matching)
4. Deduplicate against existing sheet data (`engine/dedup.py`)
5. Format rows for columns B–G (`engine/formatter.py`)
6. Insert rows if expense section is full, write to correct monthly sheet
7. When creating a new month sheet, run carryover from the previous month (`engine/carryover.py`)

The pipeline supports multiple spreadsheets (one per year) and an optional `--year` filter. Without `--year`, transactions are routed to the appropriate year's spreadsheet automatically.

## Classification: Lookup Table + Human Review

1. **Exact/prefix lookup table** built from historical sheet data (merchant name → category). Handles ~80-90% of transactions.
2. **Human review** for unknown merchants via CLI (`review` command), feeding back into the lookup table.
3. Unclassified transactions are written to the sheet with empty categories — they can be categorized later.

## Month-to-Month Carryover

`engine/carryover.py` runs automatically when the pipeline creates a new month sheet. It can also be triggered manually via the `carryover` CLI command.

### What carries over (source → destination):
- **Budget accumulated**: For CARRY-flagged lines only, positive remaining (surplus) from source → Accumulated (E) column in destination. Debts do NOT carry — they're handled by the surplus/deficit formula.
- **Savings accumulated**: Total (C) from each savings goal in source → Accumulated (F) in destination. Matched by goal name in column G; only goals present in destination get values.
- **Savings budget line**: The חסכון budget line in destination is set so that total budget = source month's income total. Formula: `savings_budget = income_total - sum(all other budget D values)`.

### Cross-year carryover:
January needs December from the previous year's spreadsheet. `get_previous_month("January")` returns `("December", -1)` where `-1` is the year offset. The pipeline resolves the correct handler from the handler dict.

## Deduplication

`engine/dedup.py` compares incoming transactions against existing sheet rows using `(date, business_name, amount)` tuples. Sheet dates are serial numbers when read with `UNFORMATTED_VALUE` — the dedup module handles conversion.

## Statement Parsing

`engine/statement_parser.py` parses Israeli CC Excel statements (.xlsx). Detects columns by header text. Extracts: date, business_name, amount, charge_date. The `filter_pending()` static method removes transactions without a charge date.

## Modules

- **`engine/`** — `StatementParser` (Excel only), `pipeline.py` (orchestrator), `dedup.py`, `formatter.py`, `carryover.py`
- **`classifier/`** — `LookupClassifier` with exact matches and prefix patterns, persisted to `.artifacts/lookup.json`
- **`sheets/`** — `SheetHandler` (Google Sheets API: auth, read, write, metadata, row insertion, marker detection). `cli.py` is the CLI.
- **`config/`** — Settings loaded from `.secrets/config.json`. Config maps years to spreadsheet IDs: `{"spreadsheets": {"2025": "id1", "2026": "id2"}, "credentials_file": "..."}`
- **`scripts/`** — One-time migration scripts
