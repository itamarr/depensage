"""
Transaction deduplication.

Compares incoming transactions against existing sheet data to avoid
writing duplicate rows.
"""

import pandas as pd

from depensage.sheets.sheet_utils import SheetUtils


def deduplicate(new_transactions, existing_rows):
    """Remove transactions that already exist in the sheet.

    Args:
        new_transactions: DataFrame with columns [date, business_name, amount].
        existing_rows: List of rows from SheetHandler.read_expense_rows(),
                       each a list: [business_name, notes, subcategory,
                       amount, category, date].

    Returns:
        DataFrame containing only non-duplicate transactions.
    """
    if new_transactions is None or new_transactions.empty:
        return new_transactions

    existing_keys = set()
    for row in existing_rows:
        if len(row) < 6 or not row[5]:
            continue
        biz = (row[0] or "").strip()
        amount_str = (row[3] or "").replace("₪", "").replace(",", "").strip()
        try:
            amount = f"{float(amount_str):.2f}"
        except (ValueError, TypeError):
            continue
        date = SheetUtils.parse_date(row[5])
        if not date:
            continue
        date_str = date.strftime("%Y-%m-%d")
        existing_keys.add((date_str, biz, amount))

    mask = []
    for _, tx in new_transactions.iterrows():
        date_str = tx["date"].strftime("%Y-%m-%d")
        biz = str(tx["business_name"]).strip()
        amount = f"{float(tx['amount']):.2f}"
        key = (date_str, biz, amount)
        mask.append(key not in existing_keys)

    return new_transactions[mask].reset_index(drop=True)
