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
