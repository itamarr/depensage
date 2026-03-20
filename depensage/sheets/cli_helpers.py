"""
Shared CLI helpers: authentication, handler creation, category fetching.
"""

import os
import sys

from depensage.config.settings import (
    load_settings, get_spreadsheet_entry, get_entries_for_year, get_all_years,
)
from depensage.sheets.spreadsheet_handler import SheetHandler


MONTH_SHEETS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _authenticate_handler(spreadsheet_id, credentials):
    """Create and authenticate a SheetHandler."""
    handler = SheetHandler(spreadsheet_id)
    if not handler.authenticate(credentials):
        print("Authentication failed.", file=sys.stderr)
        sys.exit(1)
    return handler


def _resolve_for_year(year, settings, credentials):
    """Resolve a single handler for a year.

    If multiple spreadsheets are configured for the year, uses the one
    marked as default. If no default, prompts the user to choose.

    Returns authenticated SheetHandler.
    """
    entries = get_entries_for_year(year, settings)
    if not entries:
        available_years = get_all_years(settings)
        raise ValueError(
            f"No spreadsheet configured for year {year}. "
            f"Available years: {available_years}"
        )
    if len(entries) == 1:
        _, entry = entries[0]
        return _authenticate_handler(entry["id"], credentials)

    # Multiple entries — check for default
    defaults = [(k, e) for k, e in entries if e.get("default")]
    if len(defaults) == 1:
        _, entry = defaults[0]
        return _authenticate_handler(entry["id"], credentials)

    # No default or multiple defaults — ask user
    print(f"Multiple spreadsheets for year {year}:")
    for i, (k, e) in enumerate(entries, 1):
        default_tag = " (default)" if e.get("default") else ""
        print(f"  {i}. {k}{default_tag}")
    while True:
        try:
            choice = input(f"Choose [1-{len(entries)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(entries):
                _, entry = entries[idx]
                return _authenticate_handler(entry["id"], credentials)
        except (ValueError, EOFError):
            pass
        print("Invalid choice.")


def get_handler(args):
    """Get a single SheetHandler.

    Resolution order:
    1. --spreadsheet-id (direct override)
    2. --spreadsheet (config key lookup)
    3. Auto-select if only one entry in config
    4. Error if ambiguous
    """
    settings = load_settings()
    credentials = args.credentials or settings["credentials_file"]
    credentials = os.path.abspath(credentials)

    if getattr(args, "spreadsheet_id", None):
        return _authenticate_handler(args.spreadsheet_id, credentials)

    key = getattr(args, "spreadsheet", None)
    spreadsheets = settings["spreadsheets"]

    if key:
        entry = get_spreadsheet_entry(key, settings)
        return _authenticate_handler(entry["id"], credentials)

    if len(spreadsheets) == 1:
        entry = next(iter(spreadsheets.values()))
        return _authenticate_handler(entry["id"], credentials)

    available = ", ".join(sorted(spreadsheets.keys()))
    print(f"Multiple spreadsheets configured ({available}). "
          f"Use --spreadsheet to select one.", file=sys.stderr)
    sys.exit(1)


def get_handlers_for_pipeline(args):
    """Get handlers and year filter for the pipeline.

    Returns:
        (handlers, year_filter) where:
        - handlers: dict {year (int): SheetHandler} or single SheetHandler
        - year_filter: int (filter transactions to this year) or None (all years)
    """
    settings = load_settings()
    credentials = args.credentials or settings["credentials_file"]
    credentials = os.path.abspath(credentials)

    if getattr(args, "spreadsheet_id", None):
        return _authenticate_handler(args.spreadsheet_id, credentials), None

    key = getattr(args, "spreadsheet", None)
    spreadsheets = settings["spreadsheets"]

    if key:
        entry = get_spreadsheet_entry(key, settings)
        year = entry.get("year")
        if year is None:
            print(
                f"Error: Spreadsheet '{key}' has no 'year' field. "
                f"Cannot determine which year to process.",
                file=sys.stderr,
            )
            sys.exit(1)

        handlers = {year: _authenticate_handler(entry["id"], credentials)}

        # Load previous year for carryover
        prev_key = getattr(args, "prev_spreadsheet", None)
        prev_year = year - 1
        if prev_key:
            prev_entry = get_spreadsheet_entry(prev_key, settings)
            handlers[prev_year] = _authenticate_handler(
                prev_entry["id"], credentials
            )
        else:
            prev_entries = get_entries_for_year(prev_year, settings)
            if len(prev_entries) == 1:
                _, prev_entry = prev_entries[0]
                handlers[prev_year] = _authenticate_handler(
                    prev_entry["id"], credentials
                )
            elif len(prev_entries) > 1:
                # Multiple — use default or ask
                defaults = [
                    (k, e) for k, e in prev_entries if e.get("default")
                ]
                if len(defaults) == 1:
                    _, prev_entry = defaults[0]
                    handlers[prev_year] = _authenticate_handler(
                        prev_entry["id"], credentials
                    )
                # else: skip — user can specify --prev-spreadsheet

        return handlers, year

    # No --spreadsheet: load handlers for all configured years
    all_years = get_all_years(settings)
    handlers = {}
    for year in all_years:
        handlers[year] = _resolve_for_year(year, settings, credentials)

    return handlers, None


def fetch_categories(handler):
    """Fetch category -> subcategory list mapping from the Categories sheet."""
    values = handler.get_sheet_values("Categories", "A1:N30")
    if not values:
        print("Failed to read Categories sheet.", file=sys.stderr)
        sys.exit(1)

    categories = {}
    header = values[0]
    for col_idx, cat_name in enumerate(header):
        cat_name = cat_name.strip()
        if not cat_name:
            continue
        subcats = []
        for row in values[1:]:
            if col_idx < len(row) and row[col_idx].strip():
                subcats.append(row[col_idx].strip())
        categories[cat_name] = subcats

    return categories


def find_prefix_groups(merchants, min_prefix_len=5, min_group_size=2):
    """Find groups of names sharing a common prefix.

    Returns list of {"prefix": str, "merchants": [str]}.
    """
    names = list(merchants)
    used = set()
    groups = []

    for i, name_a in enumerate(names):
        if name_a in used:
            continue
        cluster = [name_a]
        prefix = name_a
        for name_b in names[i + 1:]:
            if name_b in used:
                continue
            common = os.path.commonprefix([prefix, name_b])
            if len(common) >= min_prefix_len:
                prefix = common
                cluster.append(name_b)

        if len(cluster) >= min_group_size:
            groups.append({
                "prefix": prefix.rstrip(),
                "merchants": cluster,
            })
            used.update(cluster)

    return groups
