"""
Shared CLI helpers: authentication, handler creation, category fetching.
"""

import os
import sys

from depensage.config.settings import load_settings, get_spreadsheet_id
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


def get_handler(args):
    """Get a single SheetHandler for the specified year (or the only configured one)."""
    settings = load_settings()
    credentials = args.credentials or settings["credentials_file"]
    credentials = os.path.abspath(credentials)

    if args.spreadsheet_id:
        return _authenticate_handler(args.spreadsheet_id, credentials)

    year = getattr(args, "year", None)
    spreadsheets = settings["spreadsheets"]

    if year:
        spreadsheet_id = get_spreadsheet_id(year, settings)
    elif len(spreadsheets) == 1:
        spreadsheet_id = next(iter(spreadsheets.values()))
    else:
        available = ", ".join(sorted(spreadsheets.keys()))
        print(f"Multiple spreadsheets configured ({available}). "
              f"Use --year to select one.", file=sys.stderr)
        sys.exit(1)

    return _authenticate_handler(spreadsheet_id, credentials)


def get_handlers_for_pipeline(args):
    """Get a dict of {year: SheetHandler} for the pipeline.

    If --year is specified, returns only that year's handler (plus previous
    year if configured, for January carryover).
    Otherwise returns handlers for all configured years.
    """
    settings = load_settings()
    credentials = args.credentials or settings["credentials_file"]
    credentials = os.path.abspath(credentials)

    if args.spreadsheet_id:
        return _authenticate_handler(args.spreadsheet_id, credentials)

    spreadsheets = settings["spreadsheets"]
    year = getattr(args, "year", None)

    if year:
        year_int = int(year)
        handlers = {year_int: _authenticate_handler(
            get_spreadsheet_id(year, settings), credentials
        )}
        prev_year = str(year_int - 1)
        if prev_year in spreadsheets:
            handlers[year_int - 1] = _authenticate_handler(
                spreadsheets[prev_year], credentials
            )
        return handlers

    handlers = {}
    for y, sid in spreadsheets.items():
        handlers[int(y)] = _authenticate_handler(sid, credentials)
    return handlers


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
