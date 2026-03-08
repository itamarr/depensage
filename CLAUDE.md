# CLAUDE.md

DepenSage automates household expense tracking in a Hebrew Google Sheets spreadsheet. Parses Israeli CC statements, classifies transactions, deduplicates, and writes to the correct monthly sheet.

## Development

```bash
source .venv/bin/activate          # Python 3.12, managed with uv
uv pip install -e .                # Install for development
python -m unittest discover        # Run all tests
flake8 depensage/ && black depensage/  # Lint and format
```

## CLI

```bash
# Main pipeline
python -m depensage.sheets.cli process statement1.xlsx [statement2.xlsx ...]
python -m depensage.sheets.cli --year 2026 process statement.xlsx

# Review unknown merchants
python -m depensage.sheets.cli review statement.xlsx

# Lookup table management
python -m depensage.sheets.cli build-lookup [--output PATH]
python -m depensage.sheets.cli consolidate-patterns

# Manual carryover
python -m depensage.sheets.cli carryover December January --source-year 2025 --dest-year 2026

# Sheet inspection
python -m depensage.sheets.cli list-sheets
python -m depensage.sheets.cli read <sheet> <range>
python -m depensage.sheets.cli formulas <sheet> <range>
python -m depensage.sheets.cli metadata
```

Override defaults with `--spreadsheet-id` and `--credentials` flags.

## Architecture

**3-layer design**: core library (no I/O, no prompts) → CLI (dev/testing) → web app (end product). All business logic in core modules; CLI and web app are thin wrappers.

**Modules**: `engine/` (parser, pipeline, dedup, formatter, carryover), `classifier/` (lookup-based), `sheets/` (Google Sheets API + CLI), `config/` (settings), `scripts/` (migrations).

## Key Conventions

- Tests: `unittest` in `<module>/tests/`, Google Sheets API mocked
- Data flow: pandas DataFrames with columns `date`, `business_name`, `amount`, `category`, `subcategory`
- Spreadsheet is in Hebrew; dates are serial numbers when read with `UNFORMATTED_VALUE`
- Secrets in `.secrets/` (gitignored), artifacts in `.artifacts/` (gitignored)
- Never hardcode spreadsheet IDs, credentials, or PII in committed code
- Commit early and often, succinct one-line messages
- **Use subagents when reading sheet data** to avoid context bloat

## Detailed Documentation

For deeper context on specific topics, read these as needed:
- **[docs/spreadsheet-structure.md](docs/spreadsheet-structure.md)** — column layout, section markers, budget/savings/income sections, meta sheets, category list
- **[docs/pipeline.md](docs/pipeline.md)** — pipeline steps, classification, carryover logic, dedup, module details
- **[docs/future.md](docs/future.md)** — remaining pain points, planned features, design considerations
