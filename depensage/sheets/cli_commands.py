"""
Non-interactive CLI commands: sheet inspection, pipeline, carryover, lookup management.
"""

import json
import os
import sys
from collections import Counter

from depensage.classifier.cc_lookup import Classification, LookupClassifier, DEFAULT_LOOKUP_PATH
from depensage.sheets.cli_helpers import (
    get_handler, get_handlers_for_pipeline, MONTH_SHEETS,
)


def cmd_list_sheets(args):
    handler = get_handler(args)
    sheets = handler.get_sheet_metadata()
    if not sheets:
        print("No sheets found or failed to fetch metadata.")
        return
    for sheet in sheets:
        props = sheet["properties"]
        print(f"  {props['title']}  (id={props['sheetId']}, "
              f"rows={props['gridProperties']['rowCount']}, "
              f"cols={props['gridProperties']['columnCount']})")


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


def cmd_build_lookup(args):
    handler = get_handler(args)
    output_path = args.output or DEFAULT_LOOKUP_PATH

    merchant_categories: dict[str, list[tuple[str, str]]] = {}

    for month in MONTH_SHEETS:
        values = handler.get_sheet_values(month, "B2:G130")
        if not values:
            continue

        for row in values:
            if len(row) < 5:
                continue
            business_name = row[0].strip() if row[0] else ""
            subcategory = row[2].strip() if len(row) > 2 and row[2] else ""
            category = row[4].strip() if len(row) > 4 and row[4] else ""

            if not business_name or not category:
                continue

            if category in ("קטגוריה", "תת קטגוריה"):
                continue

            merchant_categories.setdefault(business_name, []).append(
                (category, subcategory)
            )

    exact = {}
    for merchant, pairs in merchant_categories.items():
        most_common = Counter(pairs).most_common(1)[0][0]
        exact[merchant] = {
            "category": most_common[0],
            "subcategory": most_common[1],
        }

    classifier = LookupClassifier(lookup_path=output_path)
    classifier.exact = {
        name: Classification(category=entry["category"], subcategory=entry["subcategory"])
        for name, entry in exact.items()
    }
    classifier.save()

    print(f"Built lookup table with {len(exact)} merchants → {os.path.abspath(output_path)}")


def cmd_process(args):
    """Process CC and bank statement files through the automated pipeline."""
    from depensage.engine.pipeline import run_pipeline
    from depensage.classifier.bank_lookup import BankLookupClassifier
    from depensage.classifier.income_lookup import IncomeLookupClassifier

    handlers = get_handlers_for_pipeline(args)
    classifier = LookupClassifier()
    bank_classifier = BankLookupClassifier()
    income_classifier = IncomeLookupClassifier()
    year = int(args.year) if args.year else None

    paths = args.statements
    for p in paths:
        if not os.path.exists(p):
            print(f"File not found: {p}", file=sys.stderr)
            sys.exit(1)

    staged = run_pipeline(
        paths, handlers, classifier, year=year,
        bank_classifier=bank_classifier,
        income_classifier=income_classifier,
    )

    print(f"\nPipeline staged:")
    print(staged.summary())

    if not staged.has_writes():
        print("\nNothing to write (all duplicates or empty).")
        return

    # Fetch categories with subcategories for XLSX highlighting
    categories_with_subcats = set()
    try:
        # Use any handler to read the Categories sheet
        if isinstance(handlers, dict):
            any_handler = next(iter(handlers.values()))
        else:
            any_handler = handlers
        cats = fetch_categories(any_handler)
        categories_with_subcats = {
            cat for cat, subcats in cats.items() if subcats
        }
    except Exception:
        pass  # gracefully degrade — highlight all empty subcats

    # Export XLSX for review
    xlsx_path = staged.export_xlsx(
        categories_with_subcats=categories_with_subcats
    )
    print(f"\nReview staged changes: {xlsx_path}")

    auto_confirm = getattr(args, "auto_confirm", False)
    if auto_confirm:
        result = staged.commit(handlers)
    else:
        try:
            confirm = input("\nWrite to spreadsheet? [y/N/e] ")
        except EOFError:
            confirm = "n"
        choice = confirm.strip().lower()
        if choice == "e":
            year_flag = f"--year {args.year} " if args.year else ""
            print(f"\nEdit the staged file, then commit:")
            print(f"  python -m depensage.sheets.cli {year_flag}commit {xlsx_path}")
            return
        if choice != "y":
            print("Aborted.")
            return
        result = staged.commit(handlers)

    print(f"\nCommitted:")
    for mr in result.months:
        parts = [f"{mr.written} written, {mr.duplicates} duplicates"]
        if mr.income_written or mr.income_duplicates:
            parts.append(
                f"{mr.income_written} income, {mr.income_duplicates} income dupes"
            )
        print(f"  {mr.month} {mr.year}: {', '.join(parts)}")


def cmd_commit(args):
    """Commit an edited staged XLSX to the spreadsheet."""
    from depensage.engine.staging import import_staged_xlsx
    from depensage.engine.lookup_updater import apply_lookup_updates
    from depensage.classifier.bank_lookup import BankLookupClassifier
    from depensage.classifier.income_lookup import IncomeLookupClassifier

    xlsx_path = args.xlsx
    if not os.path.exists(xlsx_path):
        print(f"File not found: {xlsx_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading staged XLSX: {xlsx_path}")
    stages, changes = import_staged_xlsx(xlsx_path)

    if not stages:
        print("No data found in XLSX.")
        return

    # Show what will be committed
    total_expenses = sum(len(s.new_expenses) for s in stages.values())
    total_income = sum(len(s.new_income) for s in stages.values())
    print(f"\nStaged: {total_expenses} expenses, {total_income} income")
    for ws_name, stage in stages.items():
        parts = []
        if stage.new_expenses:
            parts.append(f"{len(stage.new_expenses)} expenses")
        if stage.new_income:
            parts.append(f"{len(stage.new_income)} income")
        if parts:
            print(f"  {stage.month} {stage.year}: {', '.join(parts)}")

    # Show changes
    if changes:
        print(f"\nCategory changes ({len(changes)}):")
        for c in changes:
            source_tag = c.source.upper()
            old_cat = c.old_category or "(empty)"
            old_sub = c.old_subcategory or "(empty)"
            new_cat = c.new_category or "(empty)"
            new_sub = c.new_subcategory or "(empty)"
            if c.row_type == "income":
                print(f"  [{source_tag:>6}] {c.lookup_key}: {old_cat} → {new_cat}")
            else:
                cat_change = f"{old_cat} → {new_cat}" if old_cat != new_cat else new_cat
                sub_change = f"{old_sub} → {new_sub}" if old_sub != new_sub else ""
                detail = cat_change + (f" / {sub_change}" if sub_change else "")
                print(f"  [{source_tag:>6}] {c.lookup_key}: {detail}")

        # Prompt for lookup updates
        try:
            choice = input(
                f"\nUpdate lookup for all {len(changes)} change(s)? [y/n/r(eview)] "
            )
        except EOFError:
            choice = "n"

        choice = choice.strip().lower()
        if choice == "y":
            cc_cls = LookupClassifier()
            bank_cls = BankLookupClassifier()
            income_cls = IncomeLookupClassifier()
            updated = apply_lookup_updates(changes, cc_cls, bank_cls, income_cls)
            if updated:
                print(f"Updated lookup tables: {', '.join(updated)}")
        elif choice == "r":
            cc_cls = LookupClassifier()
            bank_cls = BankLookupClassifier()
            income_cls = IncomeLookupClassifier()
            for c in changes:
                source_tag = c.source.upper()
                old_display = c.old_category or "(empty)"
                new_display = c.new_category or "(empty)"
                print(f"\n  [{source_tag}] {c.lookup_key}: {old_display} → {new_display}")
                try:
                    ans = input("  Update lookup? [y/n] ").strip().lower()
                except EOFError:
                    ans = "n"
                if ans == "y" and c.new_category:
                    apply_lookup_updates([c], cc_cls, bank_cls, income_cls)
                    print("  → Updated")
        else:
            print("Skipping lookup updates.")
    else:
        print("\nNo category changes detected.")

    # Commit to spreadsheet
    handlers = get_handlers_for_pipeline(args)

    # Re-derive write coordinates from the live sheet
    from depensage.sheets.spreadsheet_handler import SECTION_MARKERS

    if not isinstance(handlers, dict):
        single = handlers
        get_h = lambda y: single
    else:
        get_h = lambda y: handlers[y]

    for ws_name, stage in stages.items():
        handler = get_h(stage.year)

        # Expenses
        if stage.new_expenses:
            first_empty = handler.find_first_empty_expense_row(stage.month)
            marker_row = handler.find_section_marker(stage.month, "budget")
            if marker_row is None:
                print(f"Warning: No budget marker in {stage.month}, skipping expenses")
                stage.new_expenses = []
                continue
            rows_needed = len(stage.new_expenses)
            last_data_row = marker_row - 2
            available = last_data_row - first_empty + 1
            insert_needed = max(0, rows_needed - available)

            stage.expense_start_row = first_empty
            stage.expense_insert_needed = insert_needed

        # Income
        if stage.new_income:
            first_empty = handler.find_first_empty_income_row(stage.month)
            if first_empty is None:
                print(f"Warning: No income insertion point in {stage.month}")
                stage.new_income = []
                continue
            savings_marker = handler.find_section_marker(stage.month, "savings")
            if savings_marker is None:
                print(f"Warning: No savings marker in {stage.month}")
                stage.new_income = []
                continue
            last_income_data = savings_marker - 3
            available = last_income_data - first_empty + 1
            insert_needed = max(0, len(stage.new_income) - available)

            stage.income_start_row = first_empty
            stage.income_insert_needed = insert_needed

    # Build a StagedPipelineResult from the imported stages and commit
    from depensage.engine.staging import StagedPipelineResult
    staged = StagedPipelineResult()
    for ws_name, stage in stages.items():
        key = (stage.month, stage.year)
        staged.month_stages[key] = stage

    if not staged.has_writes():
        print("\nNothing to write.")
        return

    try:
        confirm = input("\nWrite to spreadsheet? [y/N] ")
    except EOFError:
        confirm = "n"
    if confirm.strip().lower() != "y":
        print("Aborted.")
        return

    result = staged.commit(handlers)
    print(f"\nCommitted:")
    for mr in result.months:
        parts = [f"{mr.written} written"]
        if mr.income_written:
            parts.append(f"{mr.income_written} income")
        print(f"  {mr.month} {mr.year}: {', '.join(parts)}")


def cmd_verify(args):
    """Verify CC charges against bank lump sums."""
    from depensage.engine.bank_parser import detect_bank_transcript, parse_bank_transcript
    from depensage.engine.verification import verify_cc_charges, format_verification_report
    from depensage.sheets.cli_helpers import get_handlers_for_pipeline

    path = args.statement
    if not detect_bank_transcript(path):
        print(f"Error: {path} is not a bank transcript")
        return

    bank_result = parse_bank_transcript(path)
    if not bank_result or not bank_result.cc_lump_sums:
        print("No CC lump sums found in bank transcript.")
        return

    handlers = get_handlers_for_pipeline(args)
    year = int(args.year) if args.year else None

    if isinstance(handlers, dict):
        if year is None:
            print("Error: --year is required when multiple spreadsheets are configured.")
            return
        handler = handlers[year]
        prev_handler = handlers.get(year - 1)
    else:
        handler = handlers
        prev_handler = None

    result = verify_cc_charges(
        handler, bank_result.cc_lump_sums, year or 2026,
        prev_year_handler=prev_handler,
    )

    print("\nCC Verification Report:")
    print(format_verification_report(result))
    if result.all_matched:
        print("\n  All billing cycles match!")
    else:
        print(f"\n  {sum(1 for c in result.cycles if not c.matched)} "
              f"cycle(s) have mismatches.")


def cmd_carryover(args):
    """Manually run carryover from one month to another."""
    from depensage.engine.carryover import run_carryover
    from depensage.sheets.cli_helpers import _authenticate_handler

    from depensage.config.settings import load_settings, get_spreadsheet_id

    settings = load_settings()
    credentials = args.credentials or settings["credentials_file"]
    credentials = os.path.abspath(credentials)

    source_year = int(args.source_year or args.year or 0)
    dest_year = int(args.dest_year or args.year or 0)
    if not source_year or not dest_year:
        print("Specify --year or both --source-year and --dest-year.", file=sys.stderr)
        sys.exit(1)

    source_sid = get_spreadsheet_id(str(source_year), settings)
    dest_sid = get_spreadsheet_id(str(dest_year), settings)

    source_handler = _authenticate_handler(source_sid, credentials)
    dest_handler = _authenticate_handler(dest_sid, credentials)

    if not source_handler.sheet_exists(args.source):
        print(f"Source sheet '{args.source}' not found.", file=sys.stderr)
        sys.exit(1)
    if not dest_handler.sheet_exists(args.dest):
        print(f"Destination sheet '{args.dest}' not found.", file=sys.stderr)
        sys.exit(1)

    result = run_carryover(source_handler, args.source, dest_handler, args.dest)

    print(f"\nCarryover {args.source} ({source_year}) → {args.dest} ({dest_year}):")
    print(f"  Budget lines:  {result['budget_lines']}")
    print(f"  Savings lines: {result['savings_lines']}")
    if result['income_total'] is not None:
        print(f"  Source income:  {result['income_total']:,.2f}")
    print(f"  Savings budget: {'set' if result['savings_budget_set'] else 'not set'}")


def cmd_consolidate_patterns(args):
    """Find duplicate exact entries that should be prefix patterns."""
    classifier = LookupClassifier()
    groups = classifier.consolidate_patterns()

    if not groups:
        print("No pattern candidates found.")
        return

    print(f"\nFound {len(groups)} pattern candidate(s):\n")

    applied = 0
    for group in groups:
        print(f"  Prefix: {group['prefix']}*")
        print(f"  Category: {group['category']}" +
              (f" / {group['subcategory']}" if group['subcategory'] else ""))
        print(f"  Merchants ({len(group['merchants'])}):")
        for m in group["merchants"]:
            print(f"    - {m}")

        while True:
            choice = input("\n  Convert to pattern? (y)es / (n)o / (q)uit: ").strip().lower()
            if choice in ('y', 'n', 'q'):
                break
            print("  Invalid choice.")

        if choice == 'q':
            break
        if choice == 'y':
            classifier.apply_pattern_group(group)
            applied += 1
            print(f"  → Converted to pattern: {group['prefix']}*")
        print()

    print(f"Applied {applied} pattern(s). Lookup table updated.")
