"""
Row formatter for writing transactions to Google Sheets.

Formats a DataFrame of transactions into the 7-column layout
used by monthly expense sheets (columns B–H).
"""

import pandas as pd

from depensage.sheets.sheet_utils import SheetUtils


def format_for_sheet(transactions):
    """Format transactions into 7-column rows matching sheet columns B–H.

    Each row: [business_name, notes, subcategory, amount, category, date, status]

    Status is determined from the charge_date column (the date the CC
    company debits the bank): if charge_date falls in the same month as
    the transaction → "CC" (charged this cycle), otherwise → "" (pending,
    charged next cycle). If charge_date is missing, falls back to empty.

    For unclassified transactions, category and subcategory are empty strings.
    Date formatted as MM/DD/YYYY.

    Args:
        transactions: DataFrame with columns date, business_name, amount,
                      and optionally category, subcategory, charge_date.

    Returns:
        List of 7-element lists.
    """
    if transactions is None or transactions.empty:
        return []

    has_charge_date = "charge_date" in transactions.columns

    rows = []
    for _, tx in transactions.iterrows():
        date_str = SheetUtils.format_date_for_sheet(tx["date"])
        amount_str = f"{tx['amount']:.2f}"
        category = tx.get("category", "") or ""
        subcategory = tx.get("subcategory", "") or ""
        business_name = tx.get("business_name", "") or ""

        # Determine CC vs pending from charge_date
        if has_charge_date and pd.notna(tx.get("charge_date")):
            charge_month = tx["charge_date"].month
            charge_year = tx["charge_date"].year
            tx_month = tx["date"].month
            tx_year = tx["date"].year
            status = "CC" if (charge_year, charge_month) == (tx_year, tx_month) else ""
        else:
            status = ""

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


def format_bank_expenses_for_sheet(transactions):
    """Format bank expense transactions into 7-column rows matching B–H.

    Each row: [business_name, notes, subcategory, amount, category, date, "BANK"]

    All bank expenses are marked "BANK" (always charged to bank directly).

    Args:
        transactions: DataFrame with columns date, amount,
                      and optionally category, subcategory, business_name.

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
        rows.append([
            tx["action"],              # B
            tx.get("details", ""),     # C (notes)
            subcategory,               # D
            amount_str,                # E
            category,                  # F
            date_str,                  # G
            "BANK",                    # H
        ])

    return rows


def format_income_for_sheet(income_transactions):
    """Format income transactions into 4-column rows matching sheet columns D–G.

    Each row: [comments, amount, category, date]

    Args:
        income_transactions: DataFrame with columns date, amount,
                             and optionally category, comments.

    Returns:
        List of 4-element lists.
    """
    if income_transactions is None or income_transactions.empty:
        return []

    rows = []
    for _, tx in income_transactions.iterrows():
        date_str = SheetUtils.format_date_for_sheet(tx["date"])
        amount_str = f"{tx['amount']:.2f}"
        category = tx.get("category", "") or ""
        comments = tx.get("comments", "") or ""

        rows.append([
            comments,    # D
            amount_str,  # E
            category,    # F
            date_str,    # G
        ])

    return rows
