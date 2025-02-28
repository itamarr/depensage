"""
Credit card statement parser for the DepenSage engine.

This module handles parsing and preprocessing of credit card statement CSV files.
"""

import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class StatementParser:
    """Parser for credit card statement CSV files."""

    def __init__(self):
        """Initialize the statement parser."""
        pass

    def parse_statement(self, file_path, encoding='utf-8-sig'):
        """
        Parse a credit card statement CSV file.

        Args:
            file_path: Path to the CSV file.
            encoding: File encoding (default: utf-8-sig for Hebrew support).

        Returns:
            DataFrame with parsed data or None if failed.
        """
        try:
            # Read the CSV file
            df = pd.read_csv(file_path, encoding=encoding, header=1)

            # Keep only rows until the last non-empty date
            last_non_empty_idx = df.iloc[:, 0].last_valid_index()
            if last_non_empty_idx is not None:
                df = df.iloc[:last_non_empty_idx + 1]

            # Extract relevant columns (date, business name, amount)
            date_col = df.columns[0]  # First column is date
            business_col = df.columns[1]  # Second column is business name
            amount_col = df.columns[2]  # Third column is amount

            df = df[[date_col, business_col, amount_col]]

            # Rename columns for easier access
            df.columns = ['date', 'business_name', 'amount']

            # Clean amount (remove commas, convert to float)
            df['amount'] = df['amount'].astype(str).str.replace(',', '').astype(float)

            # Convert date to datetime format (DD/MM/YY)
            df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y')

            logger.info(f"Parsed {len(df)} transactions from {file_path}")
            return df

        except Exception as e:
            logger.error(f"Failed to parse credit card statement '{file_path}': {e}")
            return None

    def merge_statements(self, *dataframes):
        """
        Merge multiple statement dataframes and sort by date.

        Args:
            *dataframes: DataFrames to merge.

        Returns:
            Merged and sorted DataFrame or None if failed.
        """
        try:
            # Filter out None values
            valid_dfs = [df for df in dataframes if df is not None and not df.empty]

            if not valid_dfs:
                logger.warning("No valid dataframes to merge")
                return None

            merged = pd.concat(valid_dfs, ignore_index=True)
            merged = merged.sort_values(by='date')
            logger.info(f"Merged {len(merged)} transactions from {len(valid_dfs)} statements")
            return merged
        except Exception as e:
            logger.error(f"Failed to merge statements: {e}")
            return None

    def filter_new_transactions(self, transactions_df, last_date):
        """
        Filter transactions newer than the last updated date.

        Args:
            transactions_df: DataFrame with transactions.
            last_date: Last updated date.

        Returns:
            DataFrame with new transactions.
        """
        if transactions_df is None or transactions_df.empty:
            return None

        new_transactions = transactions_df[transactions_df['date'] > last_date]
        logger.info(f"Found {len(new_transactions)} new transactions after {last_date}")
        return new_transactions

    def format_transactions_for_sheet(self, classified_transactions):
        """
        Format classified transactions for adding to the Google sheet.

        Args:
            classified_transactions: DataFrame with classified transactions.

        Returns:
            List of formatted transactions.
        """
        if classified_transactions is None or classified_transactions.empty:
            return []

        formatted = []

        for _, transaction in classified_transactions.iterrows():
            # Format date to MM/DD/YYYY
            date = transaction['date'].strftime('%m/%d/%Y')

            # Format amount
            amount = f"{transaction['amount']:.2f}"

            # Get categories
            category = transaction.get('predicted_category', '')
            subcategory = transaction.get('predicted_subcategory', '')

            formatted.append([
                '',  # Column A (empty)
                '',  # Column B (empty)
                '',  # Column C (notes, empty initially)
                subcategory,  # Column D (subcategory)
                amount,  # Column E (amount)
                category,  # Column F (category)
                date,  # Column G (date)
            ])

        return formatted
