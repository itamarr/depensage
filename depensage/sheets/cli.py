"""
DepenSage CLI entry point.

Usage:
    python -m depensage.sheets.cli <command> [options]
"""

import argparse
import sys

from depensage.classifier.cc_lookup import DEFAULT_LOOKUP_PATH
from depensage.sheets.cli_commands import (
    cmd_list_sheets, cmd_read, cmd_formulas, cmd_metadata,
    cmd_build_lookup, cmd_process, cmd_carryover, cmd_consolidate_patterns,
)
from depensage.sheets.cli_review import (
    cmd_review, cmd_review_bank, cmd_review_income,
)


def main():
    parser = argparse.ArgumentParser(description="DepenSage Sheets CLI")
    parser.add_argument("--spreadsheet-id", help="Google Spreadsheet ID (overrides config)")
    parser.add_argument("--credentials", help="Path to service account JSON key")
    parser.add_argument("--year", help="Year to operate on (e.g. 2026)")

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

    review_parser = subparsers.add_parser(
        "review", help="Classify unknown merchants from a CC statement"
    )
    review_parser.add_argument("statement", help="Path to CC statement file")

    review_bank_parser = subparsers.add_parser(
        "review-bank", help="Classify unknown bank expense actions"
    )
    review_bank_parser.add_argument(
        "statement", help="Path to bank transcript file"
    )

    review_income_parser = subparsers.add_parser(
        "review-income", help="Classify unknown income actions"
    )
    review_income_parser.add_argument(
        "statement", help="Path to bank transcript file"
    )

    process_parser = subparsers.add_parser(
        "process", help="Process CC and bank statements through the automated pipeline"
    )
    process_parser.add_argument(
        "statements", nargs="+",
        help="Path(s) to CC statement and/or bank transcript file(s)",
    )

    carryover_parser = subparsers.add_parser(
        "carryover", help="Run month-to-month carryover"
    )
    carryover_parser.add_argument("source", help="Source month (e.g. December)")
    carryover_parser.add_argument("dest", help="Destination month (e.g. January)")
    carryover_parser.add_argument("--source-year", help="Source year (default: --year)")
    carryover_parser.add_argument("--dest-year", help="Destination year (default: --year)")

    subparsers.add_parser(
        "consolidate-patterns",
        help="Find duplicate exact entries that should be prefix patterns",
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
        "review": cmd_review,
        "review-bank": cmd_review_bank,
        "review-income": cmd_review_income,
        "process": cmd_process,
        "carryover": cmd_carryover,
        "consolidate-patterns": cmd_consolidate_patterns,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
