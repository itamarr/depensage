"""
Interactive CLI review commands for classifying unknown transactions.

Covers CC merchants, bank expenses, and income transactions.
"""

import os
import sys

from depensage.classifier.cc_lookup import LookupClassifier
from depensage.sheets.cli_helpers import (
    get_handler, fetch_categories, find_prefix_groups,
)


# --- Prompt helpers ---

def prompt_category(merchant, amount, date, categories, allow_back=False):
    """Prompt user to classify a CC merchant.

    Returns (category, subcategory), None to skip, 'quit', or 'back'.
    """
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

    category, subcategory = cat_list[idx], ""
    subcategory = _prompt_subcategory(categories[category])
    return (category, subcategory)


def prompt_prefix_group(group, categories):
    """Prompt user to classify a prefix group.

    Returns (category, subcategory), None to skip, or 'quit'.
    """
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
    subcategory = _prompt_subcategory(categories[category])
    return (category, subcategory)


def prompt_bank_expense(action, amount, date, details, categories, allow_back=False):
    """Prompt user to classify a bank expense.

    Returns (category, subcategory), None, 'quit', or 'back'.
    """
    cat_list = list(categories.keys())

    print(f"\n  Action:  {action}")
    print(f"  Details: {details}")
    print(f"  Amount:  {amount}  Date: {date}")
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
    subcategory = _prompt_subcategory(categories[category])

    return (category, subcategory)


def prompt_income(action, amount, date, details, allow_back=False):
    """Prompt user to classify an income transaction.

    Returns (category, comments), None, 'quit', or 'back'.
    Income categories are free-text (not from the expense categories sheet).
    """
    print(f"\n  Action:  {action}")
    print(f"  Details: {details}")
    print(f"  Amount:  {amount}  Date: {date}")
    print()
    print(f"    {'s':>2s}. Skip")
    if allow_back:
        print(f"    {'b':>2s}. Back (undo previous)")
    print(f"    {'q':>2s}. Quit review")

    choice = input("\n  Category (free text, or s/b/q): ").strip()
    if choice.lower() == 's':
        return None
    if choice.lower() == 'b' and allow_back:
        return "back"
    if choice.lower() == 'q':
        return "quit"

    category = choice
    comments = input(f"  Comments/description: ").strip()
    return (category, comments)


def _prompt_subcategory(subcats):
    """Prompt for subcategory selection. Returns subcategory string or ''."""
    if not subcats:
        return ""

    print(f"\n  Subcategories:")
    for i, sub in enumerate(subcats, 1):
        print(f"    {i:2d}. {sub}")
    print(f"    {'0':>2s}. (none)")

    while True:
        choice = input("\n  Subcategory #: ").strip()
        try:
            idx = int(choice)
            if idx == 0:
                return ""
            if 1 <= idx <= len(subcats):
                return subcats[idx - 1]
        except ValueError:
            pass
        print("  Invalid choice, try again.")


# --- Review commands ---

def cmd_review(args):
    """Review unknown CC merchants."""
    from depensage.engine.statement_parser import StatementParser

    handler = get_handler(args)
    categories = fetch_categories(handler)

    parser = StatementParser()
    transactions = parser.parse_statement(args.statement)
    if transactions is None or transactions.empty:
        print("No transactions found in statement.")
        return

    classifier = LookupClassifier()
    result = classifier.classify(transactions)

    print(f"\nClassified: {len(result.classified)}/{len(transactions)}")
    print(f"Unknown: {len(result.unclassified)}")

    if result.unclassified.empty:
        print("Nothing to review!")
        return

    unique_names = result.unclassified["business_name"].unique().tolist()

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

    remaining = [n for n in unique_names if n not in grouped_names]
    remaining_items = []
    for name in remaining:
        row = result.unclassified[result.unclassified["business_name"] == name].iloc[0]
        remaining_items.append((name, row))

    if not remaining_items:
        print(f"\nAll merchants handled via prefix groups. Lookup table updated.")
        return

    print(f"\nRemaining individual merchants: {len(remaining_items)}")
    history = []
    i = 0

    while i < len(remaining_items):
        name, row = remaining_items[i]
        date_str = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])

        choice = prompt_category(name, row["amount"], date_str, categories, allow_back=i > 0)

        if choice == "quit":
            break
        if choice == "back":
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


def cmd_review_bank(args):
    """Review unknown bank expense transactions."""
    from depensage.engine.bank_parser import detect_bank_transcript, parse_bank_transcript
    from depensage.classifier.bank_lookup import BankLookupClassifier

    handler = get_handler(args)
    categories = fetch_categories(handler)

    path = args.statement
    if not detect_bank_transcript(path):
        print(f"File does not appear to be a bank transcript: {path}",
              file=sys.stderr)
        sys.exit(1)

    bank_result = parse_bank_transcript(path)
    if not bank_result or bank_result.expenses.empty:
        print("No bank expenses found in transcript.")
        return

    classifier = BankLookupClassifier()
    result = classifier.classify(bank_result.expenses)

    print(f"\nBank expenses: {len(bank_result.expenses)}")
    print(f"Classified: {len(result.classified)}")
    print(f"Unknown: {len(result.unclassified)}")

    if result.unclassified.empty:
        print("Nothing to review!")
        return

    unique_actions = result.unclassified["action"].unique().tolist()

    prefix_groups = find_prefix_groups(unique_actions)
    grouped_actions = set()

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
            classifier.add_pattern(
                group["prefix"], category, subcategory
            )
            grouped_actions.update(group["merchants"])
            print(f"  → Saved pattern: {group['prefix']}* → {category}" +
                  (f" / {subcategory}" if subcategory else ""))

        if quit_requested:
            print(f"\nLookup table updated.")
            return

    remaining = [a for a in unique_actions if a not in grouped_actions]
    remaining_items = []
    for action in remaining:
        row = result.unclassified[
            result.unclassified["action"] == action
        ].iloc[0]
        remaining_items.append((action, row))

    if not remaining_items:
        print(f"\nAll actions handled via prefix groups. Lookup table updated.")
        return

    print(f"\nRemaining individual actions: {len(remaining_items)}")
    history = []
    i = 0

    while i < len(remaining_items):
        action, row = remaining_items[i]
        date_str = (row["date"].strftime("%Y-%m-%d")
                    if hasattr(row["date"], "strftime") else str(row["date"]))
        details = str(row.get("details", ""))

        choice = prompt_bank_expense(
            action, row["amount"], date_str, details, categories,
            allow_back=i > 0,
        )

        if choice == "quit":
            break
        if choice == "back":
            prev_action, _, _ = history.pop()
            classifier.remove_exact(prev_action)
            print(f"  ← Undid: {prev_action}")
            i -= 1
            continue
        if choice is None:
            i += 1
            continue

        category, subcategory = choice
        classifier.add_exact(action, category, subcategory)
        history.append((action, category, subcategory))
        print(f"  → Saved: {action} → {category}" +
              (f" / {subcategory}" if subcategory else ""))
        i += 1

    reviewed = len(grouped_actions) + len(history)
    print(f"\nReviewed {reviewed} actions. Bank lookup table updated.")


def cmd_review_income(args):
    """Review unknown income transactions."""
    from depensage.engine.bank_parser import detect_bank_transcript, parse_bank_transcript
    from depensage.classifier.income_lookup import IncomeLookupClassifier

    path = args.statement
    if not detect_bank_transcript(path):
        print(f"File does not appear to be a bank transcript: {path}",
              file=sys.stderr)
        sys.exit(1)

    bank_result = parse_bank_transcript(path)
    if not bank_result or bank_result.income.empty:
        print("No income transactions found in transcript.")
        return

    classifier = IncomeLookupClassifier()
    result = classifier.classify(bank_result.income)

    print(f"\nIncome transactions: {len(bank_result.income)}")
    print(f"Classified: {len(result.classified)}")
    print(f"Unknown: {len(result.unclassified)}")

    if result.unclassified.empty:
        print("Nothing to review!")
        return

    unique_actions = result.unclassified["action"].unique().tolist()

    prefix_groups = find_prefix_groups(unique_actions)
    grouped_actions = set()

    if prefix_groups:
        print(f"\nFound {len(prefix_groups)} prefix group(s):")
        quit_requested = False
        for group in prefix_groups:
            print(f"\n  Prefix group: {group['prefix']}*")
            print(f"  Matches ({len(group['merchants'])}):")
            for m in group["merchants"][:5]:
                print(f"    - {m}")
            if len(group["merchants"]) > 5:
                print(f"    ... and {len(group['merchants']) - 5} more")

            choice = input("\n  Category (free text, or s=skip, q=quit): ").strip()
            if choice.lower() == 'q':
                quit_requested = True
                break
            if choice.lower() == 's':
                continue
            category = choice
            comments = input("  Comments/description: ").strip()
            classifier.add_pattern(group["prefix"], category, comments)
            grouped_actions.update(group["merchants"])
            print(f"  → Saved pattern: {group['prefix']}* → {category}"
                  + (f" ({comments})" if comments else ""))

        if quit_requested:
            print(f"\nLookup table updated.")
            return

    remaining = [a for a in unique_actions if a not in grouped_actions]
    remaining_items = []
    for action in remaining:
        row = result.unclassified[
            result.unclassified["action"] == action
        ].iloc[0]
        remaining_items.append((action, row))

    if not remaining_items:
        print(f"\nAll actions handled via prefix groups. Lookup table updated.")
        return

    print(f"\nRemaining individual actions: {len(remaining_items)}")
    history = []
    i = 0

    while i < len(remaining_items):
        action, row = remaining_items[i]
        date_str = (row["date"].strftime("%Y-%m-%d")
                    if hasattr(row["date"], "strftime") else str(row["date"]))
        details = str(row.get("details", ""))

        choice = prompt_income(
            action, row["amount"], date_str, details, allow_back=i > 0,
        )

        if choice == "quit":
            break
        if choice == "back":
            prev_action, _, _ = history.pop()
            classifier.remove_exact(prev_action)
            print(f"  ← Undid: {prev_action}")
            i -= 1
            continue
        if choice is None:
            i += 1
            continue

        category, comments = choice
        classifier.add_exact(action, category, comments)
        history.append((action, category, comments))
        print(f"  → Saved: {action} → {category}"
              + (f" ({comments})" if comments else ""))
        i += 1

    reviewed = len(grouped_actions) + len(history)
    print(f"\nReviewed {reviewed} actions. Income lookup table updated.")
