# Coding Conventions

## File Size & Structure

- Keep files small to medium (~200-250 lines) for human readability
- The main flow of a function should be clear at a glance — large code sections should be encapsulated as methods/functions so the high-level logic reads like an outline
- When a file grows too large, break it into smaller modules grouped by cohesive responsibility
- When a function's body grows past ~30-40 lines with distinct logical steps, extract helpers

## Architecture

- **3-layer design**: core library (no I/O, no prompts) → CLI (dev/testing) → web app (end product)
- All business logic in core modules. CLI and web app are thin wrappers.
- No routine LLM calls — deterministic logic only
- Pipeline runs non-interactively: unknowns get logged/flagged, not blocking

## Testing

- `unittest` in `<module>/tests/`, Google Sheets API mocked
- No PII in test data — use fictional names/addresses
- Tests should not depend on external secrets or network access

## Data & Secrets

- Data flow: pandas DataFrames with columns `date`, `business_name`, `amount`, `category`, `subcategory`
- Secrets in `.secrets/` (gitignored), artifacts in `.artifacts/` (gitignored)
- Never hardcode spreadsheet IDs, credentials, or PII in committed code
