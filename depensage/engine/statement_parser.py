"""
Credit card statement parser for the DepenSage engine.

Handles Excel (.xlsx) statement files from Israeli CC providers.
"""

import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Hebrew header keywords for column detection
_CHARGE_DATE_HEADER = "מועד"
_COMMENTS_HEADER = "הערות"

# Hebrew text indicating a transaction is still being processed
IN_PROCESS_TEXT = "עסקה בקליטה"


class StatementParser:
    """Parser for credit card statement files (Excel)."""

    def parse_statement(self, file_path):
        """
        Parse a credit card statement Excel file.

        Extracts date, business_name, amount, and (if present) charge_date
        columns, dropping rows with missing essential data.

        Returns:
            DataFrame with columns [date, business_name, amount] (and
            optionally charge_date), or None if failed.
        """
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in (".xlsx", ".xls"):
                logger.error(f"Unsupported file format '{ext}': {file_path}")
                return None

            # Header is in row 1 (row 0 is a title row with account holder info)
            df = pd.read_excel(file_path, header=1)

            if len(df.columns) < 3:
                logger.error(f"Excel file has fewer than 3 columns: {file_path}")
                return None

            headers = list(df.columns)

            # First 3 columns are always date, business_name, amount
            result = df.iloc[:, [0, 1, 2]].copy()
            result.columns = ["date", "business_name", "amount"]

            # Look for charge_date column
            charge_date_idx = self._find_column_index(headers, _CHARGE_DATE_HEADER)
            if charge_date_idx is not None:
                result["charge_date"] = df.iloc[:, charge_date_idx].values

            # Look for comments column (used to detect in-process transactions)
            comments_idx = self._find_column_index(headers, _COMMENTS_HEADER)
            if comments_idx is not None:
                result["comments"] = df.iloc[:, comments_idx].values

            if result.empty:
                return None

            # Drop rows where essential fields are missing
            result = result.dropna(subset=["date", "business_name", "amount"])

            # Ensure date is datetime
            result["date"] = pd.to_datetime(result["date"], errors="coerce")
            result = result.dropna(subset=["date"])

            # Ensure charge_date is datetime if present
            if "charge_date" in result.columns:
                result["charge_date"] = pd.to_datetime(
                    result["charge_date"], errors="coerce"
                )

            # Ensure amount is numeric
            result["amount"] = pd.to_numeric(
                result["amount"].astype(str).str.replace(",", ""), errors="coerce"
            )
            result = result.dropna(subset=["amount"])

            result = result.reset_index(drop=True)
            logger.info(f"Parsed {len(result)} transactions from {file_path}")
            return result

        except Exception as e:
            logger.error(f"Failed to parse credit card statement '{file_path}': {e}")
            return None

    def _find_column_index(self, headers, keyword):
        """Find column index by checking if any header contains the keyword."""
        for i, h in enumerate(headers):
            if isinstance(h, str) and keyword in h:
                return i
        return None

    def merge_statements(self, *dataframes):
        """Merge multiple statement DataFrames and sort by date."""
        try:
            valid_dfs = [df for df in dataframes if df is not None and not df.empty]

            if not valid_dfs:
                logger.warning("No valid dataframes to merge")
                return None

            merged = pd.concat(valid_dfs, ignore_index=True)
            merged = merged.sort_values(by="date")
            logger.info(f"Merged {len(merged)} transactions from {len(valid_dfs)} statements")
            return merged
        except Exception as e:
            logger.error(f"Failed to merge statements: {e}")
            return None

    @staticmethod
    def filter_in_process(df):
        """Filter out in-process transactions.

        A transaction is in-process if its comments column contains
        "עסקה בקליטה" (primary signal) or if charge_date is NaT (fallback).

        If the DataFrame has neither column, returns all rows.
        """
        if df is None or df.empty:
            return df

        mask = pd.Series(True, index=df.index)

        # Primary: filter by comments column
        if "comments" in df.columns:
            mask &= df["comments"].astype(str).str.strip() != IN_PROCESS_TEXT

        # Fallback: filter by missing charge_date
        if "charge_date" in df.columns:
            mask &= df["charge_date"].notna()

        return df[mask].reset_index(drop=True)
