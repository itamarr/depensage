<p align="center">
  <img src="docs/banner.png" alt="DepenSage" width="600">
</p>

# DepenSage

*Automated household expense tracking for Google Sheets*

## Overview

DepenSage (from French "dépense" meaning expense + "sage" meaning wise) automates household expense tracking in a Hebrew Google Sheets spreadsheet. It parses Israeli credit card statements, classifies transactions using a lookup table, deduplicates against existing data, and writes everything to the correct monthly sheet.

## Features

- **Automated Pipeline**: Parse CC statements → filter pending → classify → deduplicate → write to sheet
- **Lookup-based Classification**: Exact match and prefix pattern matching against a table built from historical data
- **Interactive Review**: CLI workflow to classify unknown merchants, feeding back into the lookup table
- **Multi-year Support**: Separate spreadsheet per year, with `--year` filtering
- **Deduplication**: Safe to re-run — duplicate transactions are detected and skipped
- **Google Sheets Integration**: Creates monthly sheets from templates, inserts rows when needed
- **Hebrew Language Support**: Full support for Hebrew spreadsheets and categories
- **Excel Statement Parsing**: Processes `.xlsx` files from Israeli CC providers (Cal, etc.)

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
       "2025": "your_2025_spreadsheet_id",
       "2026": "your_2026_spreadsheet_id"
     },
     "credentials_file": ".secrets/your-credentials.json"
   }
   ```

4. **Plant section markers** (one-time, for sheets predating the template update):
   ```bash
   python scripts/plant_section_markers.py --year 2026
   ```

## Usage

### Process CC statements (main pipeline)

```bash
# Process one or more statement files
python -m depensage.sheets.cli process statement.xlsx

# Process only 2026 transactions (ignores other years in the file)
python -m depensage.sheets.cli --year 2026 process statement.xlsx
```

### Review unknown merchants

```bash
python -m depensage.sheets.cli review statement.xlsx
```

### Build lookup table from historical sheet data

```bash
python -m depensage.sheets.cli --year 2025 build-lookup
```

### Consolidate exact entries into prefix patterns

```bash
python -m depensage.sheets.cli consolidate-patterns
```

### Manual carryover between months

```bash
python -m depensage.sheets.cli carryover December January --source-year 2025 --dest-year 2026
```

### Inspect sheets (development)

```bash
python -m depensage.sheets.cli --year 2025 list-sheets
python -m depensage.sheets.cli --year 2025 read January B2:G10
python -m depensage.sheets.cli --year 2025 formulas January E130:E140
```

## Project Structure

- **`engine/`** — Statement parser (Excel), pipeline orchestrator, deduplication, row formatter, month-to-month carryover
- **`classifier/`** — Lookup-based classifier (exact + prefix patterns), persisted to `.artifacts/lookup.json`
- **`sheets/`** — Google Sheets API integration and CLI
- **`config/`** — Settings management
- **`scripts/`** — One-time migration scripts

## Development

```bash
source .venv/bin/activate

# Run all tests
python -m unittest discover

# Lint and format
flake8 depensage/
black depensage/
```

## License

MIT License

## Credits

Developed by Itamar Rosenfeld Rauch
