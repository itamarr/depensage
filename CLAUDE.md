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
# Main pipeline (CC statements and/or bank transcripts, auto-detected)
python -m depensage.sheets.cli process statement1.xlsx [bank_transcript.xlsx ...]
python -m depensage.sheets.cli -s 2026 process statement.xlsx       # specific spreadsheet
python -m depensage.sheets.cli -s 2026_dev process statement.xlsx   # dev spreadsheet

# Review unknown merchants/actions
python -m depensage.sheets.cli review statement.xlsx           # CC merchants
python -m depensage.sheets.cli review-bank transcript.xlsx     # Bank expenses
python -m depensage.sheets.cli review-income transcript.xlsx   # Income

# Lookup table management
python -m depensage.sheets.cli build-lookup [--output PATH]
python -m depensage.sheets.cli consolidate-patterns

# Manual carryover
python -m depensage.sheets.cli -s 2026 carryover December January
python -m depensage.sheets.cli carryover December January \
  --source-spreadsheet 2025 --dest-spreadsheet 2026

# CC verification against bank lump sums
python -m depensage.sheets.cli -s 2026 verify bank_transcript.xlsx

# Sheet inspection
python -m depensage.sheets.cli list-sheets
python -m depensage.sheets.cli read <sheet> <range>
python -m depensage.sheets.cli formulas <sheet> <range>
python -m depensage.sheets.cli metadata
```

Override defaults with `--spreadsheet-id`, `--credentials`, and `-s/--spreadsheet` flags.

## Web App

```bash
# Install frontend dependencies (first time only)
npm install --prefix frontend

# Build frontend
npm run build --prefix frontend
cp -r frontend/build depensage/web/static

# Run server (HTTP for local dev)
python -m depensage.web.main --no-ssl

# Run server (HTTPS — requires .secrets/cert.pem and key.pem)
python -m depensage.web.main
```

Requires `web_password` in `.secrets/config.json`. See [docs/webapp.md](docs/webapp.md) for full details.

## Architecture

**3-layer design**: core library (no I/O, no prompts) → CLI (dev/testing) → web app (end product). All business logic in core modules; CLI and web app are thin wrappers.

**Modules**: `engine/` (virtual_month, pipeline, carryover, savings_allocator, staging, parser, bank_parser, dedup, formatter, verification), `classifier/` (lookup-based: CC, bank, income), `sheets/` (Google Sheets API + CLI), `web/` (FastAPI backend + Svelte frontend), `config/` (settings), `scripts/` (migrations).

**Pipeline flow**: Sequential per-month processing via in-memory `VirtualMonth` objects. Zero writes during staging — all reads from Google Sheets, all writes at commit. Each month finalized (carryover → expenses → income → VM update → savings) before the next starts.

## Key Conventions

- Commit early and often, succinct one-line messages
- **Use subagents when reading sheet data** to avoid context bloat
- **Use `uv pip`** for package installation (not pip or python -m pip)
- Frontend built with Svelte (SvelteKit SPA) + Tailwind CSS, served as static files by FastAPI
- See [docs/coding-conventions.md](docs/coding-conventions.md) for file size, testing, and data conventions

## Detailed Documentation

For deeper context on specific topics, read these as needed:
- **[docs/spreadsheet-structure.md](docs/spreadsheet-structure.md)** — column layout, section markers, budget/savings/income sections, meta sheets, category list
- **[docs/pipeline.md](docs/pipeline.md)** — pipeline steps, classification, carryover logic, dedup, module details
- **[docs/webapp.md](docs/webapp.md)** — web app architecture, API endpoints, frontend structure, deployment
- **[docs/coding-conventions.md](docs/coding-conventions.md)** — file size, structure, testing, data & secrets rules
- **[docs/cc-verification.md](docs/cc-verification.md)** — CC billing cycle logic, verification algorithm
- **[docs/future.md](docs/future.md)** — remaining pain points, planned features, design considerations
