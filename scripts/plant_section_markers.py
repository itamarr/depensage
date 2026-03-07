#!/usr/bin/env python3
"""
Migration script to add section markers to existing month sheets.

New month sheets created from the template already have markers and
CARRY flags. This script handles older sheets that predate the template
update.

For each month sheet without markers:
1. Removes old ---END--- marker if present
2. Inserts hidden marker rows before each section:
   ---BUDGET---, ---INCOME---, ---SAVINGS---, ---RECONCILIATION---
3. Ensures column H exists and is hidden

CARRY flags in column H are managed manually in the template and
inherited by new sheets. This script does NOT set CARRY flags —
if migrating old sheets, add them manually.

Usage:
    python scripts/plant_section_markers.py --year 2026 [--dry-run] [--sheet NAME]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from depensage.config.settings import load_settings, get_spreadsheet_id
from depensage.sheets.spreadsheet_handler import SheetHandler, SECTION_MARKERS

MONTH_SHEETS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

ALL_SHEETS = MONTH_SHEETS + ["Month Template"]


def find_section_headers(handler, sheet_name):
    """Find existing section header rows by scanning columns B and C.

    Returns dict with keys: budget_header, income_label, savings_label,
    reconciliation_label, old_end_marker.
    """
    result = {}

    values_b = handler.get_sheet_values(sheet_name, "B1:B200")
    values_c = handler.get_sheet_values(sheet_name, "C1:C200")

    if not values_b and not values_c:
        return result

    max_rows = max(len(values_b or []), len(values_c or []))
    for i in range(max_rows):
        r = i + 1  # 1-based
        b_text = (values_b[i][0] if values_b and len(values_b) > i and values_b[i] else "")
        c_text = (values_c[i][0] if values_c and len(values_c) > i and values_c[i] else "")

        if b_text == "---END---":
            result["old_end_marker"] = r
        elif b_text == "כמה נשאר":
            result["budget_header"] = r

        if c_text == "הכנסות":
            result["income_label"] = r
        elif c_text == "חסכון":
            if "income_label" in result and r > result["income_label"]:
                result["savings_label"] = r
        elif c_text == "תיקונים":
            result["reconciliation_label"] = r

    return result


def detect_existing_markers(handler, sheet_name):
    """Check which section markers already exist. Returns {section: row}."""
    existing = {}
    values_b = handler.get_sheet_values(sheet_name, "B1:B200")
    if not values_b:
        return existing
    marker_to_section = {v: k for k, v in SECTION_MARKERS.items()}
    for i, row in enumerate(values_b):
        if row and row[0] in marker_to_section:
            existing[marker_to_section[row[0]]] = i + 1
    return existing


def insert_marker_row(handler, sheet_name, before_row, marker_text):
    """Insert a hidden row with marker text in column B."""
    sheet_id = handler.get_sheet_id(sheet_name)
    if sheet_id is None:
        return False

    handler.insert_rows(sheet_name, before_row, 1)

    handler.sheets_service.values().update(
        spreadsheetId=handler.spreadsheet_id,
        range=f"{sheet_name}!B{before_row}",
        valueInputOption="RAW",
        body={"values": [[marker_text]]},
    ).execute()

    body = {
        "requests": [{
            "updateDimensionProperties": {
                "range": {
                    "sheetId": sheet_id,
                    "dimension": "ROWS",
                    "startIndex": before_row - 1,
                    "endIndex": before_row,
                },
                "properties": {"hiddenByUser": True},
                "fields": "hiddenByUser",
            }
        }]
    }
    handler.sheets_service.batchUpdate(
        spreadsheetId=handler.spreadsheet_id,
        body=body,
    ).execute()

    return True


def ensure_and_hide_column_h(handler, sheet_name):
    """Expand grid to include column H if needed, then hide it."""
    sheet_id = handler.get_sheet_id(sheet_name)
    if sheet_id is None:
        return

    sheets = handler.get_sheet_metadata()
    for sheet in sheets:
        if sheet["properties"]["sheetId"] == sheet_id:
            col_count = sheet["properties"]["gridProperties"]["columnCount"]
            if col_count < 8:
                handler.sheets_service.batchUpdate(
                    spreadsheetId=handler.spreadsheet_id,
                    body={"requests": [{
                        "appendDimension": {
                            "sheetId": sheet_id,
                            "dimension": "COLUMNS",
                            "length": 8 - col_count,
                        }
                    }]},
                ).execute()
            break

    handler.sheets_service.batchUpdate(
        spreadsheetId=handler.spreadsheet_id,
        body={"requests": [{
            "updateDimensionProperties": {
                "range": {
                    "sheetId": sheet_id,
                    "dimension": "COLUMNS",
                    "startIndex": 7,  # H = index 7
                    "endIndex": 8,
                },
                "properties": {"hiddenByUser": True},
                "fields": "hiddenByUser",
            }
        }]},
    ).execute()


def process_sheet(handler, sheet_name, dry_run=False):
    """Process a single sheet: add section markers, hide column H."""
    print(f"\n  Processing {sheet_name}...")

    existing = detect_existing_markers(handler, sheet_name)
    if len(existing) == len(SECTION_MARKERS):
        print(f"    All markers already present, skipping.")
        return False

    if existing:
        print(f"    Partial markers found: {existing}. Skipping to avoid corruption.")
        print(f"    Remove existing markers manually and re-run.")
        return False

    headers = find_section_headers(handler, sheet_name)
    required = ["budget_header", "income_label", "savings_label", "reconciliation_label"]
    missing = [k for k in required if k not in headers]
    if missing:
        print(f"    Missing landmarks: {missing}. Skipping.")
        return False

    print(f"    Found: budget_header={headers['budget_header']}, "
          f"income_label={headers['income_label']}, "
          f"savings_label={headers['savings_label']}, "
          f"reconciliation_label={headers['reconciliation_label']}")

    if headers.get("old_end_marker"):
        print(f"    Old ---END--- marker at row {headers['old_end_marker']}")

    if dry_run:
        print(f"    [DRY RUN] Would insert 4 markers, hide column H")
        return True

    # Remove old ---END--- text (row stays, just clear column B)
    if headers.get("old_end_marker"):
        print(f"    Removing old ---END--- from row {headers['old_end_marker']}")
        handler.sheets_service.values().update(
            spreadsheetId=handler.spreadsheet_id,
            range=f"{sheet_name}!B{headers['old_end_marker']}",
            valueInputOption="RAW",
            body={"values": [[""]]},
        ).execute()

    # Insert markers bottom-to-top (no offset needed — each insert
    # is above all previous ones, so positions don't shift).
    markers_bottom_to_top = [
        (headers["reconciliation_label"], SECTION_MARKERS["reconciliation"]),
        (headers["savings_label"], SECTION_MARKERS["savings"]),
        (headers["income_label"], SECTION_MARKERS["income"]),
        (headers["budget_header"], SECTION_MARKERS["budget"]),
    ]

    for before_row, marker_text in markers_bottom_to_top:
        print(f"    Inserting {marker_text} before row {before_row}")
        insert_marker_row(handler, sheet_name, before_row, marker_text)

    # Ensure column H exists and is hidden
    ensure_and_hide_column_h(handler, sheet_name)

    # Verify
    markers_found = detect_existing_markers(handler, sheet_name)
    positions = sorted(markers_found.items(), key=lambda x: x[1])
    print(f"    Verification — markers: {', '.join(f'{s}={r}' for s, r in positions)}")
    expected_order = ["budget", "income", "savings", "reconciliation"]
    actual_order = [s for s, _ in positions]
    if actual_order != expected_order:
        print(f"    WARNING: markers not in expected order!")

    print(f"    Done.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Add section markers to spreadsheets")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without making changes")
    parser.add_argument("--year", required=True,
                        help="Year to operate on (e.g. 2026)")
    parser.add_argument("--sheet", help="Process only this sheet (default: all)")
    args = parser.parse_args()

    settings = load_settings()
    spreadsheet_id = get_spreadsheet_id(args.year, settings)
    credentials = os.path.abspath(settings["credentials_file"])

    handler = SheetHandler(spreadsheet_id)
    if not handler.authenticate(credentials):
        print("Authentication failed.", file=sys.stderr)
        sys.exit(1)

    sheets_to_process = [args.sheet] if args.sheet else ALL_SHEETS

    changed = 0
    for sheet_name in sheets_to_process:
        if not handler.sheet_exists(sheet_name):
            print(f"\n  {sheet_name}: SKIP (does not exist)")
            continue

        if process_sheet(handler, sheet_name, dry_run=args.dry_run):
            changed += 1

    action = "would be updated" if args.dry_run else "updated"
    print(f"\n{changed} sheet(s) {action}.")


if __name__ == "__main__":
    main()
