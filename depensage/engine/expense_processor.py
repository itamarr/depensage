"""
Main expense processing engine for DepenSage.

This module integrates the classification and spreadsheet handling
to process credit card statements and update the tracking spreadsheet.
"""

import logging

from depensage.sheets.spreadsheet_handler import SheetHandler
from depensage.engine.statement_parser import StatementParser

logger = logging.getLogger(__name__)


class ExpenseProcessor:
    """
    Main processing engine for expense tracking.

    Integrates statement parsing, classification, and spreadsheet updating.
    """

    def __init__(self, spreadsheet_id, credentials_file=None):
        self.spreadsheet_id = spreadsheet_id
        self.credentials_file = credentials_file

        self.sheet_handler = SheetHandler(spreadsheet_id)
        self.parser = StatementParser()

        if credentials_file:
            self.authenticate(credentials_file)

    def authenticate(self, credentials_file=None):
        """
        Authenticate with Google Sheets API.

        Args:
            credentials_file: Path to credentials file (uses init value if None).

        Returns:
            True if successful, False otherwise.
        """
        creds_file = credentials_file or self.credentials_file
        if not creds_file:
            logger.error("No credentials file provided")
            return False

        success = self.sheet_handler.authenticate(creds_file)
        if success:
            self.credentials_file = creds_file

        return success

    def process_statement(self, primary_file, secondary_file=None):
        """
        Process credit card statements and update the Google spreadsheet.

        Args:
            primary_file: Path to the primary credit card statement CSV.
            secondary_file: Path to a secondary credit card statement CSV (optional).

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Parse statements
            primary_data = self.parser.parse_statement(primary_file)

            statements = [primary_data]
            if secondary_file:
                secondary_data = self.parser.parse_statement(secondary_file)
                if secondary_data is not None:
                    statements.append(secondary_data)

            # Merge statements
            merged_statement = self.parser.merge_statements(*statements)

            if merged_statement is None or merged_statement.empty:
                logger.info("No transactions found in the statements")
                return True

            # TODO: classify transactions using lookup table + LLM fallback
            classified_transactions = merged_statement

            # Group transactions by month
            classified_transactions['month'] = classified_transactions['date'].dt.month
            grouped_by_month = classified_transactions.groupby('month')

            update_success = True

            for month, transactions in grouped_by_month:
                # Get or create the month sheet
                sample_date = transactions['date'].iloc[0]
                month_sheet_name = self.sheet_handler.get_or_create_month_sheet(sample_date)

                if not month_sheet_name:
                    logger.error(f"Failed to get or create sheet for month {month}")
                    update_success = False
                    continue

                # Get the last updated date for this month
                last_date = self.sheet_handler.get_last_updated_date(month_sheet_name)

                # Filter new transactions
                new_transactions = self.parser.filter_new_transactions(transactions, last_date)

                if new_transactions is None or new_transactions.empty:
                    logger.info(f"No new transactions for month {month}")
                    continue

                # Format transactions for the sheet
                formatted_transactions = self.parser.format_transactions_for_sheet(new_transactions)

                # Get the first empty row
                values = self.sheet_handler.get_sheet_values(month_sheet_name, 'A:A')
                first_empty_row = len(values) + 1 if values else 1

                # Update the month sheet
                sheet_success = self.sheet_handler.update_sheet(
                    month_sheet_name, first_empty_row, formatted_transactions
                )

                if not sheet_success:
                    update_success = False

            if update_success:
                logger.info("Successfully processed all credit card statements")
            else:
                logger.warning("Processed statements with some errors")

            return update_success

        except Exception as e:
            logger.error(f"Failed to process credit card statements: {e}")
            return False
