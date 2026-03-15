"""
Bank account transcript parser for the DepenSage engine.

Handles Excel (.xlsx) bank account statements (תנועות בחשבון) from
Israeli banks. Splits transactions into expenses (debits) and income
(credits), and extracts CC lump sum amounts for verification.
"""

import os
from dataclasses import dataclass, field
from typing import List

import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Metadata signature to identify bank transcripts (row 2, first cell)
BANK_TRANSCRIPT_SIGNATURE = "תנועות בחשבון"

# CC charge action name — these are monthly CC bill payments, not expenses
CC_CHARGE_ACTION = "כרטיסי אשראי ל"

# Bank transcript column layout (header at row 4)
_BANK_HEADER_ROW = 4
_COL_DATE = "תאריך"
_COL_ACTION = "הפעולה"
_COL_DETAILS = "פרטים"
_COL_REFERENCE = "אסמכתא"
_COL_DEBIT = "חובה"
_COL_CREDIT = "זכות"
_COL_BALANCE = "יתרה בש''ח"
_COL_BENEFICIARY = "לטובת"
_COL_PURPOSE = "עבור"


@dataclass
class CCLumpSum:
    """A single CC lump sum charge from the bank transcript."""
    date: object  # datetime
    amount: float


@dataclass
class BankParseResult:
    """Result of parsing a bank transcript."""
    expenses: pd.DataFrame  # Debits (non-CC), cols: date, action, details, amount, reference
    income: pd.DataFrame    # Credits, cols: date, action, details, amount, reference
    cc_lump_sums: List[CCLumpSum] = field(default_factory=list)
    monthly_balances: dict = field(default_factory=dict)  # (year, month) -> closing balance


def detect_bank_transcript(file_path):
    """Check if an Excel file is a bank account transcript.

    Reads the first few rows looking for the bank transcript signature.

    Returns:
        True if the file is a bank transcript, False otherwise.
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in (".xlsx", ".xls"):
            return False
        df = pd.read_excel(file_path, header=None, nrows=5)
        for row_idx in range(min(5, len(df))):
            for val in df.iloc[row_idx]:
                if isinstance(val, str) and BANK_TRANSCRIPT_SIGNATURE in val:
                    return True
        return False
    except Exception:
        return False


def parse_bank_transcript(file_path):
    """Parse a bank account transcript Excel file.

    Returns:
        BankParseResult with expenses, income, and CC lump sums,
        or None if parsing failed.
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in (".xlsx", ".xls"):
            logger.error(f"Unsupported file format '{ext}': {file_path}")
            return None

        df = pd.read_excel(file_path, header=_BANK_HEADER_ROW)

        if len(df.columns) < 6:
            logger.error(f"Bank transcript has fewer than 6 columns: {file_path}")
            return None

        headers = list(df.columns)

        # Map columns by header keywords
        col_map = {}
        for target, keyword in [
            ("date", _COL_DATE),
            ("action", _COL_ACTION),
            ("details", _COL_DETAILS),
            ("reference", _COL_REFERENCE),
            ("debit", _COL_DEBIT),
            ("credit", _COL_CREDIT),
        ]:
            idx = _find_column(headers, keyword)
            if idx is None and target in ("date", "action", "debit", "credit"):
                logger.error(
                    f"Required column '{keyword}' not found in: {file_path}"
                )
                return None
            col_map[target] = idx

        # Optional columns
        for target, keyword in [
            ("balance", _COL_BALANCE),
            ("beneficiary", _COL_BENEFICIARY),
            ("purpose", _COL_PURPOSE),
        ]:
            col_map[target] = _find_column(headers, keyword)

        # Extract and clean data
        result = pd.DataFrame()
        result["date"] = pd.to_datetime(
            df.iloc[:, col_map["date"]], errors="coerce"
        )
        result["action"] = df.iloc[:, col_map["action"]].astype(str).str.strip()

        if col_map["details"] is not None:
            result["details"] = df.iloc[:, col_map["details"]].fillna("").astype(str).str.strip()
        else:
            result["details"] = ""

        if col_map.get("reference") is not None:
            result["reference"] = df.iloc[:, col_map["reference"]].fillna("").astype(str).str.strip()
        else:
            result["reference"] = ""

        # Debit and credit amounts
        result["debit"] = pd.to_numeric(
            df.iloc[:, col_map["debit"]].astype(str).str.replace(",", ""),
            errors="coerce",
        )
        result["credit"] = pd.to_numeric(
            df.iloc[:, col_map["credit"]].astype(str).str.replace(",", ""),
            errors="coerce",
        )

        # Balance column (optional)
        if col_map.get("balance") is not None:
            result["balance"] = pd.to_numeric(
                df.iloc[:, col_map["balance"]].astype(str).str.replace(",", ""),
                errors="coerce",
            )
        else:
            result["balance"] = float("nan")

        # Drop rows without a valid date (footer/summary rows)
        result = result.dropna(subset=["date"])

        # Split: CC lump sums, expenses (other debits), income (credits)
        cc_mask = result["action"].str.contains(CC_CHARGE_ACTION, na=False)
        cc_rows = result[cc_mask]
        cc_lump_sums = [
            CCLumpSum(date=row["date"], amount=row["debit"])
            for _, row in cc_rows.iterrows()
            if pd.notna(row["debit"])
        ]

        non_cc = result[~cc_mask]
        out_cols = ["date", "action", "details", "amount", "reference"]
        empty_df = pd.DataFrame(columns=out_cols)

        # Expenses: rows with a debit value
        expense_mask = non_cc["debit"].notna() & (non_cc["debit"] > 0)
        if expense_mask.any():
            expenses = non_cc[expense_mask].copy()
            expenses["amount"] = expenses["debit"]
            expenses = expenses[out_cols].reset_index(drop=True)
        else:
            expenses = empty_df.copy()

        # Income: rows with a credit value
        income_mask = non_cc["credit"].notna() & (non_cc["credit"] > 0)
        if income_mask.any():
            income = non_cc[income_mask].copy()
            income["amount"] = income["credit"]
            income = income[out_cols].reset_index(drop=True)
        else:
            income = empty_df.copy()

        # Extract last balance per month (closing balance for the month)
        monthly_balances = _extract_monthly_balances(result)

        logger.info(
            f"Parsed bank transcript: {len(expenses)} expenses, "
            f"{len(income)} income, {len(cc_lump_sums)} CC charges, "
            f"{len(monthly_balances)} monthly balances "
            f"from {file_path}"
        )

        return BankParseResult(
            expenses=expenses,
            income=income,
            cc_lump_sums=cc_lump_sums,
            monthly_balances=monthly_balances,
        )

    except Exception as e:
        logger.error(f"Failed to parse bank transcript '{file_path}': {e}")
        return None


def _extract_monthly_balances(df):
    """Extract the last balance per (year, month) from parsed transactions.

    Uses the balance on the latest transaction date within each month.

    Args:
        df: DataFrame with 'date' and 'balance' columns.

    Returns:
        Dict mapping (year, month) to closing balance float.
    """
    if "balance" not in df.columns:
        return {}

    valid = df[df["balance"].notna() & df["date"].notna()].copy()
    if valid.empty:
        return {}

    # Group by year-month, take balance from latest date in each group
    valid["_ym"] = valid["date"].dt.to_period("M")
    balances = {}
    for period, group in valid.groupby("_ym"):
        latest = group.loc[group["date"].idxmax()]
        balances[(period.year, period.month)] = float(latest["balance"])
    return balances


def _find_column(headers, keyword):
    """Find column index by checking if any header contains the keyword."""
    for i, h in enumerate(headers):
        if isinstance(h, str) and keyword in h:
            return i
    return None
