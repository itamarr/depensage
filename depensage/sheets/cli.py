"""
CLI utility for inspecting and interacting with Google Sheets.

Usage:
    python -m depensage.sheets.cli list-sheets
    python -m depensage.sheets.cli read <sheet> <range>
    python -m depensage.sheets.cli formulas <sheet> <range>
    python -m depensage.sheets.cli metadata
    python -m depensage.sheets.cli build-lookup [--output PATH]
"""

import argparse
from collections import Counter
import json
import os
import sys

from depensage.classifier.lookup import Classification, LookupClassifier, DEFAULT_LOOKUP_PATH
from depensage.config.settings import load_settings
from depensage.sheets.spreadsheet_handler import SheetHandler


def get_handler(args):
    settings = load_settings()
    spreadsheet_id = args.spreadsheet_id or settings["spreadsheet_id"]
    credentials = args.credentials or settings["credentials_file"]
    credentials = os.path.abspath(credentials)

    handler = SheetHandler(spreadsheet_id)
    if not handler.authenticate(credentials):
        print("Authentication failed.", file=sys.stderr)
        sys.exit(1)
    return handler


def cmd_list_sheets(args):
    handler = get_handler(args)
    sheets = handler.get_sheet_metadata()
    if not sheets:
        print("No sheets found or failed to fetch metadata.")
        return
    for sheet in sheets:
        props = sheet["properties"]
        print(f"  {props['title']}  (id={props['sheetId']}, rows={props['gridProperties']['rowCount']}, cols={props['gridProperties']['columnCount']})")


def cmd_read(args):
    handler = get_handler(args)
    values = handler.get_sheet_values(args.sheet, args.range)
    if values is None:
        print("Failed to read values.", file=sys.stderr)
        sys.exit(1)
    for i, row in enumerate(values):
        print(f"  {i+1}: {row}")


def cmd_formulas(args):
    handler = get_handler(args)
    try:
        result = handler.sheets_service.values().get(
            spreadsheetId=handler.spreadsheet_id,
            range=f"{args.sheet}!{args.range}",
            valueRenderOption="FORMULA",
        ).execute()
        values = result.get("values", [])
        for i, row in enumerate(values):
            print(f"  {i+1}: {row}")
    except Exception as e:
        print(f"Failed to read formulas: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_metadata(args):
    handler = get_handler(args)
    sheets = handler.get_sheet_metadata()
    if not sheets:
        print("Failed to fetch metadata.", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(sheets, indent=2, default=str))


MONTH_SHEETS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def cmd_build_lookup(args):
    handler = get_handler(args)
    output_path = args.output or DEFAULT_LOOKUP_PATH

    # Collect (merchant -> list of (category, subcategory)) across all months
    merchant_categories: dict[str, list[tuple[str, str]]] = {}

    for month in MONTH_SHEETS:
        # Read columns B (business name), D (subcategory), F (category) — rows 2-130
        values = handler.get_sheet_values(month, "B2:G130")
        if not values:
            continue

        for row in values:
            # Row is relative to B column: B=0, C=1, D=2, E=3, F=4, G=5
            if len(row) < 5:
                continue
            business_name = row[0].strip() if row[0] else ""
            subcategory = row[2].strip() if len(row) > 2 and row[2] else ""
            category = row[4].strip() if len(row) > 4 and row[4] else ""

            if not business_name or not category:
                continue

            # Skip header rows that leaked through
            if category in ("קטגוריה", "תת קטגוריה"):
                continue

            merchant_categories.setdefault(business_name, []).append(
                (category, subcategory)
            )

    # For each merchant, pick the most frequent (category, subcategory) pair
    exact = {}
    for merchant, pairs in merchant_categories.items():
        most_common = Counter(pairs).most_common(1)[0][0]
        exact[merchant] = {
            "category": most_common[0],
            "subcategory": most_common[1],
        }

    # Load existing lookup to preserve patterns, then merge exact entries
    classifier = LookupClassifier(lookup_path=output_path)
    classifier.exact = {
        name: Classification(category=entry["category"], subcategory=entry["subcategory"])
        for name, entry in exact.items()
    }
    classifier.save()

    print(f"Built lookup table with {len(exact)} merchants → {os.path.abspath(output_path)}")


def main():
    parser = argparse.ArgumentParser(description="DepenSage Sheets CLI")
    parser.add_argument("--spreadsheet-id", help="Google Spreadsheet ID")
    parser.add_argument("--credentials", help="Path to service account JSON key")

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list-sheets", help="List all sheet tabs")

    read_parser = subparsers.add_parser("read", help="Read cell values")
    read_parser.add_argument("sheet", help="Sheet/tab name")
    read_parser.add_argument("range", help="Cell range (e.g. A1:F10)")

    formulas_parser = subparsers.add_parser("formulas", help="Read formulas")
    formulas_parser.add_argument("sheet", help="Sheet/tab name")
    formulas_parser.add_argument("range", help="Cell range (e.g. A1:F10)")

    subparsers.add_parser("metadata", help="Full sheet metadata as JSON")

    build_lookup_parser = subparsers.add_parser(
        "build-lookup", help="Build lookup table from historical sheet data"
    )
    build_lookup_parser.add_argument(
        "--output", help=f"Output path (default: {DEFAULT_LOOKUP_PATH})"
    )

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "list-sheets": cmd_list_sheets,
        "read": cmd_read,
        "formulas": cmd_formulas,
        "metadata": cmd_metadata,
        "build-lookup": cmd_build_lookup,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
