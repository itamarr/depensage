"""
Google Sheets interaction module for expense tracking.

This module handles all interactions with Google Sheets,
including authentication, reading, and writing data.
"""

import os
from datetime import datetime
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
import logging

from depensage.sheets.sheet_utils import SheetUtils, find_first_empty_row

logger = logging.getLogger(__name__)

SECTION_MARKERS = {
    "budget": "---BUDGET---",
    "income": "---INCOME---",
    "savings": "---SAVINGS---",
    "reconciliation": "---RECONCILIATION---",
}

# All marker values for scanning
_ALL_MARKERS = set(SECTION_MARKERS.values())


class SheetHandler:
    """
    Handles interaction with Google Sheets for expense tracking.
    """

    def __init__(self, spreadsheet_id):
        """
        Initialize the SheetHandler.

        Args:
            spreadsheet_id: The ID of the Google spreadsheet.
        """
        self.spreadsheet_id = spreadsheet_id
        self.sheets_service = None

    def authenticate(self, credentials_file):
        """
        Authenticate with Google Sheets API.

        Args:
            credentials_file: Path to the service account credentials JSON file.

        Returns:
            True if successful, False otherwise.
        """
        try:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_file,
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
            service = build('sheets', 'v4', credentials=credentials)
            self.sheets_service = service.spreadsheets()
            logger.info("Successfully authenticated with Google Sheets API")
            return True
        except Exception as e:
            logger.error(f"Failed to authenticate with Google Sheets API: {e}")
            return False

    def get_sheet_metadata(self):
        """
        Get metadata about all sheets in the spreadsheet.

        Returns:
            List of sheet metadata or None if failed.
        """
        try:
            if not self.sheets_service:
                logger.error("Sheets service not initialized. Call authenticate() first.")
                return None

            sheet_metadata = self.sheets_service.get(spreadsheetId=self.spreadsheet_id).execute()
            sheets = sheet_metadata.get('sheets', [])
            return sheets
        except Exception as e:
            logger.error(f"Failed to get sheet metadata: {e}")
            return None

    def sheet_exists(self, sheet_name):
        """
        Check if a sheet with the given name exists.

        Args:
            sheet_name: Name of the sheet to check.

        Returns:
            True if exists, False otherwise.
        """
        sheets = self.get_sheet_metadata()
        if not sheets:
            return False

        return any(sheet['properties']['title'] == sheet_name for sheet in sheets)

    def get_sheet_values(self, sheet_name, range_name):
        """
        Get values from a specific range in a sheet.

        Args:
            sheet_name: Name of the sheet.
            range_name: Range to fetch (e.g., 'A1:G100').

        Returns:
            Values from the specified range or None if failed.
        """
        try:
            if not self.sheets_service:
                logger.error("Sheets service not initialized. Call authenticate() first.")
                return None

            result = self.sheets_service.values().get(
                spreadsheetId=self.spreadsheet_id,
                range=f'{sheet_name}!{range_name}'
            ).execute()

            return result.get('values', [])
        except Exception as e:
            logger.error(f"Failed to get values from {sheet_name}!{range_name}: {e}")
            return None

    def get_last_updated_date(self, sheet_name):
        """
        Get the date of the last transaction in the specified month sheet.

        Args:
            sheet_name: Name of the month sheet.

        Returns:
            datetime object of the last updated date or None if failed.
        """
        try:
            # Get the date column (column G)
            values = self.get_sheet_values(sheet_name, 'G:G')

            if not values or len(values) <= 1:
                # If no values or only header, return a very old date
                return datetime(1900, 1, 1)

            # Find the last non-empty date
            date_values = [val[0] for val in values if val and val[0] and val[0] != 'תאריך']
            if not date_values:
                return datetime(1900, 1, 1)

            last_date_str = date_values[-1]

            # Parse the date
            last_date = SheetUtils.parse_date(last_date_str)
            if not last_date:
                return datetime(1900, 1, 1)

            logger.info(f"Last updated date in sheet {sheet_name}: {last_date}")
            return last_date

        except Exception as e:
            logger.error(f"Failed to get last updated date: {e}")
            # Return a very old date if there's an error
            return datetime(1900, 1, 1)

    def create_sheet_from_template(self, new_sheet_name, template_name='Month Template'):
        """
        Create a new sheet by copying a template sheet.

        Args:
            new_sheet_name: Name for the new sheet.
            template_name: Name of the template sheet to copy.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if not self.sheets_service:
                logger.error("Sheets service not initialized. Call authenticate() first.")
                return False

            # Check if the sheet already exists
            if self.sheet_exists(new_sheet_name):
                logger.info(f"Sheet '{new_sheet_name}' already exists")
                return True

            # Find the template sheet ID
            sheets = self.get_sheet_metadata()
            template_sheet_id = None

            # If template_name is None, create a blank sheet instead
            if template_name is None:
                body = {
                    'requests': [
                        {
                            'addSheet': {
                                'properties': {
                                    'title': new_sheet_name
                                }
                            }
                        }
                    ]
                }

                self.sheets_service.batchUpdate(
                    spreadsheetId=self.spreadsheet_id,
                    body=body
                ).execute()

                logger.info(f"Created new blank sheet '{new_sheet_name}'")
                return True

            # Otherwise, copy from template
            for sheet in sheets:
                if sheet['properties']['title'] == template_name:
                    template_sheet_id = sheet['properties']['sheetId']
                    break

            if template_sheet_id is None:
                logger.error(f"Template sheet '{template_name}' not found")
                return False

            # Create a new sheet as a copy of the template
            body = {
                'requests': [
                    {
                        'duplicateSheet': {
                            'sourceSheetId': template_sheet_id,
                            'insertSheetIndex': 0,  # Insert at the beginning
                            'newSheetName': new_sheet_name
                        }
                    }
                ]
            }

            self.sheets_service.batchUpdate(
                spreadsheetId=self.spreadsheet_id,
                body=body
            ).execute()

            logger.info(f"Created new sheet '{new_sheet_name}' from template")
            return True

        except Exception as e:
            logger.error(f"Failed to create sheet from template: {e}")
            return False

    def update_sheet(self, sheet_name, row_index, values):
        """
        Update values in a sheet starting at a specific row.

        Args:
            sheet_name: Name of the sheet to update.
            row_index: Starting row index (1-based).
            values: List of row values to update.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if not self.sheets_service:
                logger.error("Sheets service not initialized. Call authenticate() first.")
                return False

            if not values:
                logger.info(f"No values to update in {sheet_name}")
                return True

            body = {
                'values': values
            }

            result = self.sheets_service.values().update(
                spreadsheetId=self.spreadsheet_id,
                range=f'{sheet_name}!A{row_index}',
                valueInputOption='USER_ENTERED',
                body=body
            ).execute()

            logger.info(f"Updated {result.get('updatedCells')} cells in {sheet_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to update sheet: {e}")
            return False

    def get_or_create_month_sheet(self, date):
        """
        Get the sheet for the given date's month, creating it from template if it doesn't exist.

        Args:
            date: datetime object to get the month sheet for.

        Returns:
            Name of the month sheet or None if failed.
        """
        sheet_name = SheetUtils.get_sheet_name_for_date(date)

        if not sheet_name:
            logger.error(f"Could not determine sheet name for date {date}")
            return None

        if not self.sheet_exists(sheet_name):
            success = self.create_sheet_from_template(sheet_name)
            if not success:
                return None

        return sheet_name

    def get_sheet_id(self, sheet_name):
        """Get numeric sheet ID from metadata.

        Args:
            sheet_name: Name of the sheet.

        Returns:
            Integer sheet ID, or None if not found.
        """
        sheets = self.get_sheet_metadata()
        if not sheets:
            return None
        for sheet in sheets:
            if sheet['properties']['title'] == sheet_name:
                return sheet['properties']['sheetId']
        return None

    def find_section_marker(self, sheet_name, section):
        """Find the row containing a section marker in column B.

        Args:
            sheet_name: Name of the sheet.
            section: Section name (key in SECTION_MARKERS, e.g. "budget").

        Returns:
            1-based row number, or None if not found.
        """
        marker = SECTION_MARKERS.get(section)
        if not marker:
            logger.error(f"Unknown section: {section}")
            return None
        values = self.get_sheet_values(sheet_name, 'B1:B200')
        if not values:
            return None
        for i, row in enumerate(values):
            if row and row[0] == marker:
                return i + 1  # 1-based
        return None

    def read_expense_rows(self, sheet_name):
        """Read expense data rows (B2 to the row before the budget marker).

        The expense total row sits at (budget_marker - 1), so data rows
        end at (budget_marker - 2).

        Uses UNFORMATTED_VALUE so dates come as serial numbers and
        amounts as raw numbers rather than locale-formatted strings.

        Returns:
            List of rows, each a list [business_name, notes,
            subcategory, amount, category, date_serial]. Returns empty
            list if marker not found.
        """
        marker_row = self.find_section_marker(sheet_name, "budget")
        if not marker_row or marker_row <= 3:
            return []
        last_data_row = marker_row - 2
        try:
            result = self.sheets_service.values().get(
                spreadsheetId=self.spreadsheet_id,
                range=f'{sheet_name}!B2:G{last_data_row}',
                valueRenderOption='UNFORMATTED_VALUE',
            ).execute()
            return result.get('values', [])
        except Exception as e:
            logger.error(f"Failed to read expense rows from {sheet_name}: {e}")
            return []

    def find_first_empty_expense_row(self, sheet_name):
        """Find first empty row between row 2 and the budget marker.

        Scans column G (date) since every expense has a date.
        Data rows end at (budget_marker - 2), since (marker - 1) is the total row.

        Returns:
            1-based row number, or None if marker not found.
        """
        marker_row = self.find_section_marker(sheet_name, "budget")
        if not marker_row:
            return None
        # Data rows end at marker_row - 2 (total row is marker_row - 1)
        last_data_row = marker_row - 2
        if last_data_row < 2:
            return 2
        values = self.get_sheet_values(sheet_name, f'G2:G{last_data_row}')
        if not values:
            return 2
        # Find first empty cell
        for i, row in enumerate(values):
            if not row or not row[0]:
                return i + 2  # 1-based, offset by header row
        # All rows filled — next row after data
        return len(values) + 2

    def insert_rows(self, sheet_name, row_index, count):
        """Insert empty rows at the given position.

        Uses batchUpdate insertDimension with inheritFromBefore=True.

        Args:
            sheet_name: Name of the sheet.
            row_index: 1-based row index where rows will be inserted.
            count: Number of rows to insert.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if not self.sheets_service:
                logger.error("Sheets service not initialized.")
                return False

            sheet_id = self.get_sheet_id(sheet_name)
            if sheet_id is None:
                logger.error(f"Sheet '{sheet_name}' not found")
                return False

            body = {
                'requests': [{
                    'insertDimension': {
                        'range': {
                            'sheetId': sheet_id,
                            'dimension': 'ROWS',
                            'startIndex': row_index - 1,  # 0-based
                            'endIndex': row_index - 1 + count,
                        },
                        'inheritFromBefore': True,
                    }
                }]
            }

            self.sheets_service.batchUpdate(
                spreadsheetId=self.spreadsheet_id,
                body=body,
            ).execute()

            logger.info(f"Inserted {count} rows at row {row_index} in {sheet_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to insert rows: {e}")
            return False

    def write_expense_rows(self, sheet_name, start_row, rows):
        """Write expense rows to columns B–G.

        Args:
            sheet_name: Name of the sheet.
            start_row: 1-based row to start writing.
            rows: List of 6-element lists [B, C, D, E, F, G].

        Returns:
            True if successful, False otherwise.
        """
        try:
            if not self.sheets_service:
                logger.error("Sheets service not initialized.")
                return False

            if not rows:
                return True

            end_row = start_row + len(rows) - 1
            range_str = f'{sheet_name}!B{start_row}:G{end_row}'

            body = {'values': rows}
            result = self.sheets_service.values().update(
                spreadsheetId=self.spreadsheet_id,
                range=range_str,
                valueInputOption='USER_ENTERED',
                body=body,
            ).execute()

            logger.info(f"Wrote {result.get('updatedCells')} cells to {range_str}")
            return True

        except Exception as e:
            logger.error(f"Failed to write expense rows: {e}")
            return False

    def extract_historical_data(self):
        """
        Extract historical transaction data from all month sheets.

        Returns:
            DataFrame with historical transaction data or empty DataFrame if failed.
        """
        try:
            if not self.sheets_service:
                logger.error("Sheets service not initialized. Call authenticate() first.")
                return pd.DataFrame()

            # Get all sheets
            sheets = self.get_sheet_metadata()
            if not sheets:
                return pd.DataFrame()

            all_data = []

            for sheet in sheets:
                sheet_name = sheet['properties']['title']

                # Skip non-month sheets and special sheets
                if sheet_name == 'Categories' or sheet_name == 'Month Template':
                    continue

                # Get sheet data
                values = self.get_sheet_values(sheet_name, 'A1:G')

                if not values or len(values) <= 1:
                    continue

                # Extract data rows (skip header)
                for row in values[1:]:
                    if len(row) < 7 or not row[6]:  # Date is in column G
                        continue

                    # Extract date, amount, category, subcategory
                    date_str = row[6]
                    amount = row[4] if len(row) > 4 and row[4] else "0"
                    category = row[5] if len(row) > 5 and row[5] else ""
                    subcategory = row[3] if len(row) > 3 and row[3] else ""
                    business_name = row[1] if len(row) > 1 and row[1] else ""

                    # Clean amount
                    amount = amount.replace('₪', '').replace(',', '').strip()
                    if amount:
                        try:
                            amount = float(amount)
                        except ValueError:
                            continue
                    else:
                        continue

                    # Parse date
                    date = SheetUtils.parse_date(date_str)
                    if not date:
                        continue

                    all_data.append({
                        'date': date,
                        'amount': amount,
                        'category': category,
                        'subcategory': subcategory,
                        'business_name': business_name
                    })

            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            logger.info(f"Extracted {len(df)} historical transactions")
            return df

        except Exception as e:
            logger.error(f"Failed to extract historical data: {e}")
            return pd.DataFrame()
