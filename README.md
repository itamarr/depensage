<p align="center">
  <img src="docs/banner.png" alt="DepenSage" width="600">
</p>

# DepenSage

*Automated household expense tracking for Google Sheets*

## Overview

DepenSage (from French "dépense" meaning expense + "sage" meaning wise) automates household expense tracking in a Hebrew Google Sheets spreadsheet. It parses Israeli credit card and bank statements, classifies transactions, deduplicates against existing data, computes month-to-month carryover, allocates savings, and writes everything to the correct monthly sheet.

## Features

- **Automated Pipeline**: Parse CC & bank statements → classify → deduplicate → stage → review → commit
- **In-Memory Staging**: All computation happens in-memory via VirtualMonth objects. Google Sheets is read-only during staging — zero writes until you commit. Export to XLSX for review before writing.
- **Sequential Per-Month Processing**: Each month is fully finalized (carryover → expenses → income → savings) before the next, ensuring accurate data chains
- **Month-to-Month Carryover**: Budget surplus, savings accumulated, and savings budget automatically carry over. Works across years (Dec 2025 → Jan 2026).
- **Savings Allocation**: Auto-computes from carryover-set savings budget, deducts preset targets, allocates surplus to a configurable default goal
- **Lookup-based Classification**: Exact match and prefix pattern matching for CC merchants, bank actions, and income sources
- **Interactive Review**: Export staged XLSX → edit classifications → import with change detection → update lookup tables
- **Multi-year Support**: Separate spreadsheet per year with automatic cross-year handler resolution
- **Bank Transcript Integration**: Parses Israeli bank transcripts — expenses, income, CC lump sums, and closing balances
- **CC Verification**: Compare CC lump sums from bank against CC-tagged expenses per billing cycle
- **Deduplication**: Safe to re-run — duplicate transactions are detected and skipped
- **Hebrew Language Support**: Full support for Hebrew spreadsheets and categories

## Requirements

- Python 3.12+
- Google Sheets API service account credentials
- [uv](https://github.com/astral-sh/uv) (recommended for package management)

## Setup

1. **Google Sheets API**:
   - Create a Google Cloud project and enable the Sheets API
   - Create a service account and download the JSON key
   - Share your expense spreadsheet with the service account email

2. **Install**:
   ```bash
   git clone https://github.com/itamarr/depensage.git
   cd depensage
   uv venv && source .venv/bin/activate
   uv pip install -e .
   ```

3. **Configure** (`.secrets/config.json`, gitignored):
   ```json
   {
     "spreadsheets": {
       "2025": {"id": "spreadsheet_id_here", "year": 2025},
       "2026": {"id": "spreadsheet_id_here", "year": 2026, "default": true}
     },
     "credentials_file": ".secrets/your-credentials.json",
     "default_savings_goal": "דירה"
   }
   ```

## Usage

### Main pipeline (CC statements and/or bank transcripts)

```bash
python -m depensage.sheets.cli -s 2026 process statement.xlsx bank_transcript.xlsx
```

The pipeline stages all changes and exports an XLSX for review. You can then write directly, edit the XLSX first, or abort.

### Review unknown merchants

```bash
python -m depensage.sheets.cli review statement.xlsx           # CC merchants
python -m depensage.sheets.cli review-bank transcript.xlsx     # Bank expenses
python -m depensage.sheets.cli review-income transcript.xlsx   # Income
```

### Manual carryover between months

```bash
python -m depensage.sheets.cli -s 2026 carryover December January
python -m depensage.sheets.cli carryover December January \
  --source-spreadsheet 2025 --dest-spreadsheet 2026
```

### CC verification against bank lump sums

```bash
python -m depensage.sheets.cli -s 2026 verify bank_transcript.xlsx
```

### Lookup table management

```bash
python -m depensage.sheets.cli build-lookup
python -m depensage.sheets.cli consolidate-patterns
```

### Inspect sheets

```bash
python -m depensage.sheets.cli -s 2025 list-sheets
python -m depensage.sheets.cli -s 2025 read January B2:G10
python -m depensage.sheets.cli -s 2025 formulas January E130:E140
```

## Web App

The web app provides a visual interface for the full pipeline workflow.

### Build & Run

```bash
# Install web dependencies
uv pip install fastapi "uvicorn[standard]" python-multipart

# Build frontend (requires Node.js 18+)
npm install --prefix frontend
npm run build --prefix frontend
cp -r frontend/build depensage/web/static

# Add password to .secrets/config.json
# "web_password": "your-password"

# Run (HTTP for local dev)
python -m depensage.web.main --no-ssl

# Run (HTTPS for production)
python -m depensage.web.main
```

Access at `http://localhost:8000` or `https://<your-ip>:8000` from any device on your network.

See [docs/webapp.md](docs/webapp.md) for full architecture, API reference, and deployment details.

## Project Structure

- **`engine/`** — Core pipeline: `virtual_month.py` (in-memory month representation), `pipeline.py` (sequential orchestrator), `carryover.py` (budget/savings carry), `savings_allocator.py`, `staging.py` (XLSX export/import), `dedup.py`, `formatter.py`, `statement_parser.py`, `bank_parser.py`, `verification.py`
- **`classifier/`** — Lookup-based classifiers (CC, bank, income) with exact + prefix patterns
- **`sheets/`** — Google Sheets API integration (`spreadsheet_handler.py`) and CLI (`cli.py`, `cli_commands.py`)
- **`web/`** — FastAPI backend (`app.py`, `routers/`) serving both API and Svelte frontend
- **`config/`** — Settings management
- **`frontend/`** — Svelte/SvelteKit SPA source (builds to `web/static/`)
- **`scripts/`** — One-time migration scripts

## Development

```bash
source .venv/bin/activate
python -m unittest discover    # Run all tests (195)

# Frontend live-reload (for UI development)
npm run dev --prefix frontend  # Proxies /api to backend at :8000
```

## License

MIT License

## Credits

Developed by Itamar Rosenfeld Rauch
