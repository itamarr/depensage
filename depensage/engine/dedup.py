"""
Transaction deduplication.

Compares incoming transactions against existing sheet data to avoid
writing duplicate rows.
"""

from datetime import datetime, timedelta

import pandas as pd


# Google Sheets epoch: December 30, 1899
_SHEETS_EPOCH = datetime(1899, 12, 30)


def _parse_sheet_date(value):
    """Parse a date value from a sheet row.

    Handles both serial numbers (from UNFORMATTED_VALUE) and
    date strings like MM/DD/YYYY.

    Returns a datetime or None.
    """
    if value is None:
        return None

    # Serial number (int or float)
    if isinstance(value, (int, float)):
        try:
            return _SHEETS_EPOCH + timedelta(days=int(value))
        except (ValueError, OverflowError):
            return None

    # String date
    if isinstance(value, str):
        for fmt in ("%m/%d/%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue

    return None


def deduplicate(new_transactions, existing_rows):
    """Remove transactions that already exist in the sheet.

    Args:
        new_transactions: DataFrame with columns [date, business_name, amount].
        existing_rows: List of rows from SheetHandler.read_expense_rows(),
                       each a list: [business_name, notes, subcategory,
                       amount, category, date]. Date may be a serial number
                       (int/float) or a string.

    Returns:
        DataFrame containing only non-duplicate transactions.
    """
    if new_transactions is None or new_transactions.empty:
        return new_transactions

    existing_keys = set()
    for row in existing_rows:
        if len(row) < 6 or not row[5]:
            continue
        biz = str(row[0] or "").strip()
        try:
            amount = f"{float(row[3]):.2f}"
        except (ValueError, TypeError):
            continue
        date = _parse_sheet_date(row[5])
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


def deduplicate_income(new_income, existing_rows):
    """Remove income transactions that already exist in the sheet.

    Args:
        new_income: DataFrame with columns [date, amount, category, comments].
        existing_rows: List of rows from SheetHandler.read_income_rows(),
                       each a list: [comments, amount, category, date].
                       Date may be a serial number or string.

    Returns:
        DataFrame containing only non-duplicate income transactions.
    """
    if new_income is None or new_income.empty:
        return new_income

    existing_keys = set()
    for row in existing_rows:
        if len(row) < 4 or not row[3]:
            continue
        comments = str(row[0] or "").strip()
        try:
            amount = f"{float(row[1]):.2f}"
        except (ValueError, TypeError):
            continue
        category = str(row[2] or "").strip()
        date = _parse_sheet_date(row[3])
        if not date:
            continue
        date_str = date.strftime("%Y-%m-%d")
        existing_keys.add((date_str, amount, category, comments))

    mask = []
    for _, tx in new_income.iterrows():
        date_str = tx["date"].strftime("%Y-%m-%d")
        amount = f"{float(tx['amount']):.2f}"
        category = str(tx.get("category", "")).strip()
        comments = str(tx.get("comments", "")).strip()
        key = (date_str, amount, category, comments)
        mask.append(key not in existing_keys)

    return new_income[mask].reset_index(drop=True)
