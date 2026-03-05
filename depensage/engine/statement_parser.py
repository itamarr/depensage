"""
Credit card statement parser for the DepenSage engine.

Handles both CSV and Excel (.xlsx) formats from Israeli CC providers.
"""

import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class StatementParser:
    """Parser for credit card statement files (CSV and Excel)."""

    def parse_statement(self, file_path):
        """
        Parse a credit card statement file.

        Supports CSV and Excel (.xlsx) formats. Extracts date, business name,
        and amount columns, dropping rows with missing data.

        Returns:
            DataFrame with columns [date, business_name, amount] or None if failed.
        """
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in (".xlsx", ".xls"):
                df = self._parse_excel(file_path)
            else:
                df = self._parse_csv(file_path)

            if df is None or df.empty:
                return None

            # Drop rows where essential fields are missing
            df = df.dropna(subset=["date", "business_name", "amount"])

            # Ensure date is datetime
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])

            # Ensure amount is numeric
            df["amount"] = pd.to_numeric(
                df["amount"].astype(str).str.replace(",", ""), errors="coerce"
            )
            df = df.dropna(subset=["amount"])

            df = df.reset_index(drop=True)
            logger.info(f"Parsed {len(df)} transactions from {file_path}")
            return df

        except Exception as e:
            logger.error(f"Failed to parse credit card statement '{file_path}': {e}")
            return None

    def _parse_excel(self, file_path):
        """Parse an Excel statement file (Israeli CC format)."""
        # Header is in row 1 (row 0 is a title row with account holder info)
        df = pd.read_excel(file_path, header=1)

        # Take first 3 columns: date, business name, amount
        if len(df.columns) < 3:
            logger.error(f"Excel file has fewer than 3 columns: {file_path}")
            return None

        result = df.iloc[:, [0, 1, 2]].copy()
        result.columns = ["date", "business_name", "amount"]
        return result

    def _parse_csv(self, file_path):
        """Parse a CSV statement file."""
        df = pd.read_csv(file_path, encoding="utf-8-sig", header=1)

        if len(df.columns) < 3:
            logger.error(f"CSV file has fewer than 3 columns: {file_path}")
            return None

        result = df.iloc[:, [0, 1, 2]].copy()
        result.columns = ["date", "business_name", "amount"]
        return result

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

    def filter_new_transactions(self, transactions_df, last_date):
        """Filter transactions newer than the last updated date."""
        if transactions_df is None or transactions_df.empty:
            return None

        new_transactions = transactions_df[transactions_df["date"] > last_date]
        logger.info(f"Found {len(new_transactions)} new transactions after {last_date}")
        return new_transactions

    def format_transactions_for_sheet(self, classified_transactions):
        """Format classified transactions for adding to the Google sheet."""
        if classified_transactions is None or classified_transactions.empty:
            return []

        formatted = []

        for _, tx in classified_transactions.iterrows():
            date = tx["date"].strftime("%m/%d/%Y")
            amount = f"{tx['amount']:.2f}"
            category = tx.get("category", "")
            subcategory = tx.get("subcategory", "")
            business_name = tx.get("business_name", "")

            formatted.append([
                "",              # Column A (empty)
                business_name,   # Column B (business name)
                "",              # Column C (notes)
                subcategory,     # Column D (subcategory)
                amount,          # Column E (amount)
                category,        # Column F (category)
                date,            # Column G (date)
            ])

        return formatted
