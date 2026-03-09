"""
Row formatter for writing transactions to Google Sheets.

Formats a DataFrame of transactions into the 7-column layout
used by monthly expense sheets (columns B–H).
"""

import pandas as pd

from depensage.sheets.sheet_utils import SheetUtils

# Default CC billing day (day of month when charges are debited from bank)
DEFAULT_BILLING_DAY = 10


def format_for_sheet(transactions, billing_day=DEFAULT_BILLING_DAY):
    """Format transactions into 7-column rows matching sheet columns B–H.

    Each row: [business_name, notes, subcategory, amount, category, date, status]

    Status is "CHARGED" if the transaction date's day-of-month <= billing_day,
    empty string otherwise (pending — will be charged next billing cycle).

    For unclassified transactions, category and subcategory are empty strings.
    Date formatted as MM/DD/YYYY.

    Args:
        transactions: DataFrame with columns date, business_name, amount,
                      and optionally category, subcategory.
        billing_day: Day of month when CC charges are debited (default 10).

    Returns:
        List of 7-element lists.
    """
    if transactions is None or transactions.empty:
        return []

    rows = []
    for _, tx in transactions.iterrows():
        date_str = SheetUtils.format_date_for_sheet(tx["date"])
        amount_str = f"{tx['amount']:.2f}"
        category = tx.get("category", "") or ""
        subcategory = tx.get("subcategory", "") or ""
        business_name = tx.get("business_name", "") or ""
        status = "CHARGED" if tx["date"].day <= billing_day else ""

        rows.append([
            business_name,   # B
            "",              # C (notes)
            subcategory,     # D
            amount_str,      # E
            category,        # F
            date_str,        # G
            status,          # H
        ])

    return rows
