"""
Row formatter for writing transactions to Google Sheets.

Formats a DataFrame of transactions into the 6-column layout
used by monthly expense sheets (columns B–G).
"""

import pandas as pd

from depensage.sheets.sheet_utils import SheetUtils


def format_for_sheet(transactions):
    """Format transactions into 6-column rows matching sheet columns B–G.

    Each row: [business_name, notes, subcategory, amount, category, date]

    For unclassified transactions, category and subcategory are empty strings.
    Date formatted as MM/DD/YYYY.

    Args:
        transactions: DataFrame with columns date, business_name, amount,
                      and optionally category, subcategory.

    Returns:
        List of 6-element lists.
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

        rows.append([
            business_name,   # B
            "",              # C (notes)
            subcategory,     # D
            amount_str,      # E
            category,        # F
            date_str,        # G
        ])

    return rows
