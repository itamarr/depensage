#!/usr/bin/env python3
"""
One-time migration script to plant EXPENSE_END_MARKER in existing month sheets
and the Month Template.

For each month sheet, reads column E with FORMULA valueRenderOption to detect
the first formula row after expense data (the totals row). Writes ---END---
to column B of that row.

Usage:
    python scripts/plant_markers.py [--dry-run]
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from depensage.config.settings import load_settings
from depensage.sheets.spreadsheet_handler import SheetHandler, EXPENSE_END_MARKER

MONTH_SHEETS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

ALL_SHEETS = MONTH_SHEETS + ["Month Template"]


def find_totals_row(handler, sheet_name):
    """Find the totals row by looking for the first SUMIF/SUMIFS formula in column E.

    Returns 1-based row number, or None if not found.
    """
    try:
        result = handler.sheets_service.values().get(
            spreadsheetId=handler.spreadsheet_id,
            range=f"{sheet_name}!E1:E200",
            valueRenderOption="FORMULA",
        ).execute()
        values = result.get("values", [])

        for i, row in enumerate(values):
            if not row or not row[0]:
                continue
            cell = str(row[0])
            if cell.startswith("=") and ("SUMIF" in cell.upper() or "SUM" in cell.upper()):
                return i + 1  # 1-based
    except Exception as e:
        print(f"  Error reading formulas from {sheet_name}: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description="Plant expense end markers")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without making changes")
    args = parser.parse_args()

    settings = load_settings()
    spreadsheet_id = settings["spreadsheet_id"]
    credentials = os.path.abspath(settings["credentials_file"])

    handler = SheetHandler(spreadsheet_id)
    if not handler.authenticate(credentials):
        print("Authentication failed.", file=sys.stderr)
        sys.exit(1)

    # Check which sheets exist
    changes = []
    for sheet_name in ALL_SHEETS:
        if not handler.sheet_exists(sheet_name):
            print(f"  {sheet_name}: SKIP (does not exist)")
            continue

        # Check if marker already exists
        existing_marker = handler.find_expense_end_row(sheet_name)
        if existing_marker:
            print(f"  {sheet_name}: SKIP (marker already at row {existing_marker})")
            continue

        totals_row = find_totals_row(handler, sheet_name)
        if totals_row is None:
            print(f"  {sheet_name}: SKIP (no totals formula found)")
            continue

        changes.append((sheet_name, totals_row))
        print(f"  {sheet_name}: will place marker at row {totals_row}")

    if not changes:
        print("\nNo changes needed.")
        return

    if args.dry_run:
        print(f"\nDry run: {len(changes)} sheet(s) would be updated.")
        return

    # Confirm
    response = input(f"\nApply {len(changes)} change(s)? (y/N): ").strip().lower()
    if response != "y":
        print("Aborted.")
        return

    # Apply
    for sheet_name, row in changes:
        success = handler.sheets_service.values().update(
            spreadsheetId=handler.spreadsheet_id,
            range=f"{sheet_name}!B{row}",
            valueInputOption="RAW",
            body={"values": [[EXPENSE_END_MARKER]]},
        ).execute()
        print(f"  {sheet_name}: planted marker at row {row}")

    print(f"\nDone. {len(changes)} marker(s) planted.")


if __name__ == "__main__":
    main()
