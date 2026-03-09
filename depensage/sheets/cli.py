"""
CLI utility for inspecting and interacting with Google Sheets.

Usage:
    python -m depensage.sheets.cli list-sheets
    python -m depensage.sheets.cli read <sheet> <range>
    python -m depensage.sheets.cli formulas <sheet> <range>
    python -m depensage.sheets.cli metadata
    python -m depensage.sheets.cli build-lookup [--output PATH]
    python -m depensage.sheets.cli review <statement.csv>
"""

import argparse
from collections import Counter
import json
import os
import sys

from depensage.classifier.lookup import Classification, LookupClassifier, DEFAULT_LOOKUP_PATH
from depensage.config.settings import load_settings, get_spreadsheet_id
from depensage.sheets.spreadsheet_handler import SheetHandler


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

    If --year is specified, returns only that year's handler.
    Otherwise returns handlers for all configured years.
    """
    settings = load_settings()
    credentials = args.credentials or settings["credentials_file"]
    credentials = os.path.abspath(credentials)

    if args.spreadsheet_id:
        # Single override — use for all years
        return _authenticate_handler(args.spreadsheet_id, credentials)

    spreadsheets = settings["spreadsheets"]
    year = getattr(args, "year", None)

    if year:
        year_int = int(year)
        handlers = {year_int: _authenticate_handler(
            get_spreadsheet_id(year, settings), credentials
        )}
        # Include previous year if configured (needed for Jan carryover)
        prev_year = str(year_int - 1)
        if prev_year in spreadsheets:
            handlers[year_int - 1] = _authenticate_handler(
                spreadsheets[prev_year], credentials
            )
        return handlers

    # All years
    handlers = {}
    for y, sid in spreadsheets.items():
        handlers[int(y)] = _authenticate_handler(sid, credentials)
    return handlers


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


def prompt_category(merchant, amount, date, categories, allow_back=False):
    """Interactively prompt user to classify a merchant. Returns (category, subcategory), None to skip, 'quit', or 'back'."""
    cat_list = list(categories.keys())

    print(f"\n  Merchant: {merchant}")
    print(f"  Amount: {amount}  Date: {date}")
    print()
    for i, cat in enumerate(cat_list, 1):
        print(f"    {i:2d}. {cat}")
    print(f"    {'s':>2s}. Skip")
    if allow_back:
        print(f"    {'b':>2s}. Back (undo previous)")
    print(f"    {'q':>2s}. Quit review")

    while True:
        choice = input("\n  Category #: ").strip()
        if choice.lower() == 's':
            return None
        if choice.lower() == 'b' and allow_back:
            return "back"
        if choice.lower() == 'q':
            return "quit"
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(cat_list):
                break
        except ValueError:
            pass
        print("  Invalid choice, try again.")

    category = cat_list[idx]
    subcategory = ""

    subcats = categories[category]
    if subcats:
        print(f"\n  Subcategories for {category}:")
        for i, sub in enumerate(subcats, 1):
            print(f"    {i:2d}. {sub}")
        print(f"    {'0':>2s}. (none)")

        while True:
            choice = input("\n  Subcategory #: ").strip()
            try:
                idx = int(choice)
                if idx == 0:
                    break
                if 1 <= idx <= len(subcats):
                    subcategory = subcats[idx - 1]
                    break
            except ValueError:
                pass
            print("  Invalid choice, try again.")

    return (category, subcategory)


def find_prefix_groups(merchants, min_prefix_len=5, min_group_size=2):
    """
    Find groups of merchant names sharing a common prefix.

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


def prompt_prefix_group(group, categories):
    """Prompt user to classify a prefix group. Returns (category, subcategory), None to skip, or 'quit'."""
    print(f"\n  Prefix group: {group['prefix']}*")
    print(f"  Matches ({len(group['merchants'])}):")
    for m in group["merchants"][:5]:
        print(f"    - {m}")
    if len(group["merchants"]) > 5:
        print(f"    ... and {len(group['merchants']) - 5} more")

    cat_list = list(categories.keys())
    print()
    for i, cat in enumerate(cat_list, 1):
        print(f"    {i:2d}. {cat}")
    print(f"    {'s':>2s}. Skip (classify individually)")
    print(f"    {'q':>2s}. Quit review")

    while True:
        choice = input("\n  Category #: ").strip()
        if choice.lower() == 's':
            return None
        if choice.lower() == 'q':
            return "quit"
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(cat_list):
                break
        except ValueError:
            pass
        print("  Invalid choice, try again.")

    category = cat_list[idx]
    subcategory = ""

    subcats = categories[category]
    if subcats:
        print(f"\n  Subcategories for {category}:")
        for i, sub in enumerate(subcats, 1):
            print(f"    {i:2d}. {sub}")
        print(f"    {'0':>2s}. (none)")

        while True:
            choice = input("\n  Subcategory #: ").strip()
            try:
                idx = int(choice)
                if idx == 0:
                    break
                if 1 <= idx <= len(subcats):
                    subcategory = subcats[idx - 1]
                    break
            except ValueError:
                pass
            print("  Invalid choice, try again.")

    return (category, subcategory)


def cmd_review(args):
    from depensage.engine.statement_parser import StatementParser

    handler = get_handler(args)
    categories = fetch_categories(handler)

    # Parse the statement
    parser = StatementParser()
    transactions = parser.parse_statement(args.statement)
    if transactions is None or transactions.empty:
        print("No transactions found in statement.")
        return

    # Classify
    classifier = LookupClassifier()
    result = classifier.classify(transactions)

    print(f"\nClassified: {len(result.classified)}/{len(transactions)}")
    print(f"Unknown: {len(result.unclassified)}")

    if result.unclassified.empty:
        print("Nothing to review!")
        return

    # Deduplicate by merchant name
    unique_names = result.unclassified["business_name"].unique().tolist()

    # Phase 1: Find prefix groups among unknowns
    prefix_groups = find_prefix_groups(unique_names)
    grouped_names = set()

    if prefix_groups:
        print(f"\nFound {len(prefix_groups)} prefix group(s):")
        quit_requested = False
        for group in prefix_groups:
            choice = prompt_prefix_group(group, categories)

            if choice == "quit":
                quit_requested = True
                break
            if choice is None:
                continue

            category, subcategory = choice
            classifier.add_pattern(group["prefix"], category, subcategory)
            grouped_names.update(group["merchants"])
            print(f"  → Saved pattern: {group['prefix']}* → {category}" +
                  (f" / {subcategory}" if subcategory else ""))

        if quit_requested:
            print(f"\nLookup table updated.")
            return

    # Phase 2: Review remaining merchants one by one with undo support
    remaining = [n for n in unique_names if n not in grouped_names]
    # Build a list of (name, sample_row) for display
    remaining_items = []
    for name in remaining:
        row = result.unclassified[result.unclassified["business_name"] == name].iloc[0]
        remaining_items.append((name, row))

    if not remaining_items:
        print(f"\nAll merchants handled via prefix groups. Lookup table updated.")
        return

    print(f"\nRemaining individual merchants: {len(remaining_items)}")
    # history tracks (merchant_name, category, subcategory) for undo
    history = []
    i = 0

    while i < len(remaining_items):
        name, row = remaining_items[i]
        date_str = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])

        choice = prompt_category(name, row["amount"], date_str, categories, allow_back=i > 0)

        if choice == "quit":
            break
        if choice == "back":
            # Undo last classification
            prev_name, prev_cat, prev_sub = history.pop()
            classifier.remove_exact(prev_name)
            classifier.save()
            print(f"  ← Undid: {prev_name}")
            i -= 1
            continue
        if choice is None:
            i += 1
            continue

        category, subcategory = choice
        classifier.add_exact(name, category, subcategory)
        history.append((name, category, subcategory))
        print(f"  → Saved: {name} → {category}" +
              (f" / {subcategory}" if subcategory else ""))
        i += 1

    reviewed = len(grouped_names) + len(history)
    print(f"\nReviewed {reviewed} merchants. Lookup table updated.")


def cmd_process(args):
    """Process CC statement files through the automated pipeline."""
    from depensage.engine.pipeline import run_pipeline

    handlers = get_handlers_for_pipeline(args)
    classifier = LookupClassifier()
    year = int(args.year) if args.year else None

    paths = args.statements
    for p in paths:
        if not os.path.exists(p):
            print(f"File not found: {p}", file=sys.stderr)
            sys.exit(1)

    result = run_pipeline(paths, handlers, classifier, year=year)

    print(f"\nPipeline complete:")
    print(f"  Parsed:     {result.total_parsed}")
    print(f"  In-process: {result.in_process_skipped} (skipped)")
    print(f"  Classified: {result.classified}")
    print(f"  Unknown:    {result.unclassified}")
    print()

    for mr in result.months:
        print(f"  {mr.month} {mr.year}: {mr.written} written, {mr.duplicates} duplicates")

    if result.unclassified_merchants:
        print(f"\n  Unclassified merchants ({len(result.unclassified_merchants)}):")
        for name in result.unclassified_merchants:
            print(f"    - {name}")


def cmd_carryover(args):
    """Manually run carryover from one month to another."""
    from depensage.engine.carryover import run_carryover

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

    process_parser = subparsers.add_parser(
        "process", help="Process CC statements through the automated pipeline"
    )
    process_parser.add_argument(
        "statements", nargs="+", help="Path(s) to CC statement file(s)"
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
        "process": cmd_process,
        "carryover": cmd_carryover,
        "consolidate-patterns": cmd_consolidate_patterns,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
