"""
CLI utility for inspecting and interacting with Google Sheets.

Usage:
    python -m depensage.sheets.cli list-sheets
    python -m depensage.sheets.cli read <sheet> <range>
    python -m depensage.sheets.cli formulas <sheet> <range>
    python -m depensage.sheets.cli metadata
"""

import argparse
import json
import os
import sys

from depensage.sheets.spreadsheet_handler import SheetHandler


# Defaults - override with --spreadsheet-id and --credentials
DEFAULT_SPREADSHEET_ID = "1UI01PHhvcVrw5SkWATUMEX3kkiqxS1htL006gQn_bpk"
DEFAULT_CREDENTIALS = os.path.join(
    os.path.dirname(__file__), "..", "..", ".secrets", "depensage-0560a8b53051.json"
)


def get_handler(args):
    spreadsheet_id = args.spreadsheet_id or DEFAULT_SPREADSHEET_ID
    credentials = args.credentials or DEFAULT_CREDENTIALS
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

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "list-sheets": cmd_list_sheets,
        "read": cmd_read,
        "formulas": cmd_formulas,
        "metadata": cmd_metadata,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
