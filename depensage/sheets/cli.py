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
    cmd_build_lookup, cmd_process, cmd_commit, cmd_carryover,
    cmd_consolidate_patterns, cmd_verify, cmd_set_password,
    cmd_config_show, cmd_config_add, cmd_config_update, cmd_config_remove,
    cmd_config_set,
)
from depensage.sheets.cli_review import (
    cmd_review, cmd_review_bank, cmd_review_income,
)


def main():
    parser = argparse.ArgumentParser(description="DepenSage Sheets CLI")
    parser.add_argument("--spreadsheet-id", help="Google Spreadsheet ID (overrides config)")
    parser.add_argument("--credentials", help="Path to service account JSON key")
    parser.add_argument(
        "--spreadsheet", "-s",
        help="Spreadsheet config key (e.g. 2026, 2026_dev). "
             "Filters transactions to that spreadsheet's year.",
    )
    parser.add_argument(
        "--prev-spreadsheet",
        help="Previous year spreadsheet key for cross-year carryover "
             "(auto-detected if omitted)",
    )

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
    process_parser.add_argument(
        "-y", "--auto-confirm", action="store_true",
        help="Skip confirmation prompt and write immediately",
    )

    carryover_parser = subparsers.add_parser(
        "carryover", help="Run month-to-month carryover"
    )
    carryover_parser.add_argument("source", help="Source month (e.g. December)")
    carryover_parser.add_argument("dest", help="Destination month (e.g. January)")
    carryover_parser.add_argument(
        "--source-spreadsheet",
        help="Source spreadsheet key (default: previous year of --spreadsheet)",
    )
    carryover_parser.add_argument(
        "--dest-spreadsheet",
        help="Destination spreadsheet key (default: --spreadsheet)",
    )

    commit_parser = subparsers.add_parser(
        "commit", help="Commit a staged XLSX to the spreadsheet (no lookup review)"
    )
    commit_parser.add_argument(
        "xlsx", help="Path to the staged XLSX file"
    )

    verify_parser = subparsers.add_parser(
        "verify", help="Verify CC charges against bank lump sums"
    )
    verify_parser.add_argument(
        "statement", help="Path to bank transcript file"
    )

    subparsers.add_parser(
        "consolidate-patterns",
        help="Find duplicate exact entries that should be prefix patterns",
    )

    set_password_parser = subparsers.add_parser(
        "set-password", help="Set the web app login password"
    )
    set_password_parser.add_argument(
        "password", nargs="?",
        help="Password to set (omit to be prompted securely)",
    )

    # Config management
    subparsers.add_parser("config", help="Show current configuration")

    config_add_parser = subparsers.add_parser(
        "config-add", help="Add or update a spreadsheet entry"
    )
    config_add_parser.add_argument("key", help="Config key (e.g. 2027, 2027_dev)")
    config_add_parser.add_argument("spreadsheet_id", help="Google Spreadsheet ID")
    config_add_parser.add_argument(
        "--year", type=int, help="Year this spreadsheet covers"
    )
    config_add_parser.add_argument(
        "--default", action="store_true",
        help="Set as default for its year",
    )

    config_update_parser = subparsers.add_parser(
        "config-update", help="Update an existing spreadsheet entry"
    )
    config_update_parser.add_argument("key", help="Config key to update")
    config_update_parser.add_argument(
        "--id", dest="spreadsheet_id",
        help="New Google Spreadsheet ID",
    )
    config_update_parser.add_argument(
        "--year", type=int, help="New year",
    )
    config_update_parser.add_argument(
        "--default", action="store_true",
        help="Set as default for its year",
    )

    config_rm_parser = subparsers.add_parser(
        "config-remove", help="Remove a spreadsheet entry"
    )
    config_rm_parser.add_argument("key", help="Config key to remove")

    config_set_parser = subparsers.add_parser(
        "config-set", help="Set a config value (credentials_file, default_savings_goal)"
    )
    config_set_parser.add_argument(
        "key", help="Config key (credentials_file or default_savings_goal)"
    )
    config_set_parser.add_argument("value", help="Value to set")

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
        "commit": cmd_commit,
        "carryover": cmd_carryover,
        "verify": cmd_verify,
        "consolidate-patterns": cmd_consolidate_patterns,
        "set-password": cmd_set_password,
        "config": cmd_config_show,
        "config-add": cmd_config_add,
        "config-update": cmd_config_update,
        "config-remove": cmd_config_remove,
        "config-set": cmd_config_set,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
