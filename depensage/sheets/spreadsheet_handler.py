"""
Google Sheets interaction module for expense tracking.

This module handles all interactions with Google Sheets,
including authentication, reading, and writing data.

Sheet data is cached on first access to minimize API calls.
The cache is updated in-place after writes/inserts.
"""

import os
from dataclasses import dataclass, field
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


@dataclass
class _CachedSheet:
    """Cached data for a single sheet tab."""
    raw_data: list[list]  # Full A:H data, 0-indexed rows
    markers: dict[str, int]  # section name -> 1-based row number


@dataclass
class _SheetCache:
    """Internal cache for sheet metadata and data."""
    metadata: list | None = None  # Sheet metadata (names, IDs, row counts)
    sheets: dict[str, _CachedSheet] = field(default_factory=dict)


class SheetHandler:
    """
    Handles interaction with Google Sheets for expense tracking.

    Sheet data is cached lazily on first access per sheet.
    Writes update the cache in-place to keep it consistent.
    """

    def __init__(self, spreadsheet_id):
        """
        Initialize the SheetHandler.

        Args:
            spreadsheet_id: The ID of the Google spreadsheet.
        """
        self.spreadsheet_id = spreadsheet_id
        self.sheets_service = None
        self._cache = _SheetCache()

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

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _ensure_metadata_cached(self):
        """Fetch and cache sheet metadata if not already cached."""
        if self._cache.metadata is not None:
            return
        try:
            if not self.sheets_service:
                logger.error("Sheets service not initialized. Call authenticate() first.")
                return
            result = self.sheets_service.get(
                spreadsheetId=self.spreadsheet_id
            ).execute()
            self._cache.metadata = result.get('sheets', [])
            logger.debug("Cached sheet metadata (%d sheets)", len(self._cache.metadata))
        except Exception as e:
            logger.error(f"Failed to get sheet metadata: {e}")

    def _ensure_sheet_cached(self, sheet_name):
        """Fetch full A:H data for *sheet_name* and build marker index."""
        if sheet_name in self._cache.sheets:
            return
        self._ensure_metadata_cached()
        if self._cache.metadata is None:
            return

        # Determine row count from metadata
        row_count = None
        for sheet in self._cache.metadata:
            if sheet['properties']['title'] == sheet_name:
                row_count = sheet['properties']['gridProperties']['rowCount']
                break
        if row_count is None:
            logger.error(f"Sheet '{sheet_name}' not found in metadata")
            return

        try:
            result = self.sheets_service.values().get(
                spreadsheetId=self.spreadsheet_id,
                range=f'{sheet_name}!A1:H{row_count}',
                valueRenderOption='UNFORMATTED_VALUE',
            ).execute()
            raw = result.get('values', [])
        except Exception as e:
            logger.error(f"Failed to cache sheet '{sheet_name}': {e}")
            return

        # Build marker index by scanning column B (index 1)
        markers = {}
        for i, row in enumerate(raw):
            if len(row) > 1 and isinstance(row[1], str) and row[1] in _ALL_MARKERS:
                # Reverse-lookup section name
                for sec_name, sec_marker in SECTION_MARKERS.items():
                    if row[1] == sec_marker:
                        markers[sec_name] = i + 1  # 1-based
                        break
        self._cache.sheets[sheet_name] = _CachedSheet(raw_data=raw, markers=markers)
        logger.debug("Cached sheet '%s': %d rows, markers=%s", sheet_name, len(raw), markers)

    def _get_cached(self, sheet_name):
        """Return _CachedSheet for *sheet_name*, or None."""
        self._ensure_sheet_cached(sheet_name)
        return self._cache.sheets.get(sheet_name)

    def invalidate_cache(self, sheet_name=None):
        """Drop cached data (useful after structural changes).

        Args:
            sheet_name: If given, drop only that sheet. Otherwise drop all.
        """
        if sheet_name:
            self._cache.sheets.pop(sheet_name, None)
        else:
            self._cache.metadata = None
            self._cache.sheets.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_sheet_metadata(self):
        """
        Get metadata about all sheets in the spreadsheet.

        Returns:
            List of sheet metadata or None if failed.
        """
        self._ensure_metadata_cached()
        return self._cache.metadata

    def sheet_exists(self, sheet_name):
        """
        Check if a sheet with the given name exists (uses cached metadata).

        Args:
            sheet_name: Name of the sheet to check.

        Returns:
            True if exists, False otherwise.
        """
        self._ensure_metadata_cached()
        if not self._cache.metadata:
            return False
        return any(
            sheet['properties']['title'] == sheet_name
            for sheet in self._cache.metadata
        )

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

                self._cache.metadata = None
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

            # Invalidate metadata cache since a new sheet was added
            self._cache.metadata = None

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
        """Get numeric sheet ID from cached metadata.

        Args:
            sheet_name: Name of the sheet.

        Returns:
            Integer sheet ID, or None if not found.
        """
        self._ensure_metadata_cached()
        if not self._cache.metadata:
            return None
        for sheet in self._cache.metadata:
            if sheet['properties']['title'] == sheet_name:
                return sheet['properties']['sheetId']
        return None

    def find_section_marker(self, sheet_name, section):
        """Find the row containing a section marker in column B (from cache).

        Args:
            sheet_name: Name of the sheet.
            section: Section name (key in SECTION_MARKERS, e.g. "budget").

        Returns:
            1-based row number, or None if not found.
        """
        if section not in SECTION_MARKERS:
            logger.error(f"Unknown section: {section}")
            return None
        cached = self._get_cached(sheet_name)
        if not cached:
            return None
        return cached.markers.get(section)

    def read_expense_rows(self, sheet_name):
        """Read expense data rows (B2 to the row before the budget marker).

        The expense total row sits at (budget_marker - 1), so data rows
        end at (budget_marker - 2).

        Returns from cache. Values are UNFORMATTED (dates as serial
        numbers, amounts as raw numbers).

        Returns:
            List of rows, each a list [business_name, notes,
            subcategory, amount, category, date_serial]. Returns empty
            list if marker not found.
        """
        cached = self._get_cached(sheet_name)
        if not cached:
            return []
        marker_row = cached.markers.get("budget")
        if not marker_row or marker_row <= 3:
            return []
        # raw_data is 0-indexed. Row 2 = index 1, last data = marker_row-2 = index marker_row-3
        start_idx = 1  # row 2
        end_idx = marker_row - 2  # inclusive, so slice to marker_row-1
        rows = cached.raw_data[start_idx:end_idx]
        # Extract columns B:G (indices 1-6)
        return [row[1:7] if len(row) > 1 else [] for row in rows]

    def read_expense_rows_with_status(self, sheet_name):
        """Read expense data rows including status column (B2:H, from cache).

        Same as read_expense_rows but includes column H (status: CC/BANK/empty).

        Returns:
            List of rows, each a list [business_name, notes, subcategory,
            amount, category, date_serial, status]. Returns empty list if
            marker not found.
        """
        cached = self._get_cached(sheet_name)
        if not cached:
            return []
        marker_row = cached.markers.get("budget")
        if not marker_row or marker_row <= 3:
            return []
        start_idx = 1
        end_idx = marker_row - 2
        rows = cached.raw_data[start_idx:end_idx]
        # Extract columns B:H (indices 1-7)
        return [row[1:8] if len(row) > 1 else [] for row in rows]

    def find_first_empty_expense_row(self, sheet_name):
        """Find first empty row between row 2 and the budget marker (from cache).

        Scans column G (date) since every expense has a date.
        Data rows end at (budget_marker - 2), since (marker - 1) is the total row.

        Returns:
            1-based row number, or None if marker not found.
        """
        cached = self._get_cached(sheet_name)
        if not cached:
            return None
        marker_row = cached.markers.get("budget")
        if not marker_row:
            return None
        last_data_row = marker_row - 2
        if last_data_row < 2:
            return 2
        # Scan column G (index 6) in rows 2..last_data_row
        for i in range(1, last_data_row):  # 0-indexed: row 2 = index 1
            row = cached.raw_data[i] if i < len(cached.raw_data) else []
            date_val = row[6] if len(row) > 6 else None
            if not date_val:
                return i + 1  # 1-based
        # All rows filled
        return last_data_row + 1

    def insert_rows(self, sheet_name, row_index, count):
        """Insert empty rows at the given position.

        Uses batchUpdate insertDimension with inheritFromBefore=True.
        Updates cache in-place (splices empty rows, shifts markers).

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

            # Update cache: splice empty rows and shift markers
            cached = self._cache.sheets.get(sheet_name)
            if cached:
                insert_idx = row_index - 1  # 0-based
                empty_rows = [[] for _ in range(count)]
                cached.raw_data[insert_idx:insert_idx] = empty_rows
                # Shift markers that are at or after the insertion point
                for sec, mrow in list(cached.markers.items()):
                    if mrow >= row_index:
                        cached.markers[sec] = mrow + count

            logger.info(f"Inserted {count} rows at row {row_index} in {sheet_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to insert rows: {e}")
            return False

    def write_expense_rows(self, sheet_name, start_row, rows):
        """Write expense rows to columns B–H.

        Updates cache in-place after a successful API write.

        Args:
            sheet_name: Name of the sheet.
            start_row: 1-based row to start writing.
            rows: List of 7-element lists [B, C, D, E, F, G, H].

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
            range_str = f'{sheet_name}!B{start_row}:H{end_row}'

            body = {'values': rows}
            result = self.sheets_service.values().update(
                spreadsheetId=self.spreadsheet_id,
                range=range_str,
                valueInputOption='USER_ENTERED',
                body=body,
            ).execute()

            # Patch cache: write values into columns B-H (indices 1-7)
            cached = self._cache.sheets.get(sheet_name)
            if cached:
                for i, row_data in enumerate(rows):
                    idx = start_row - 1 + i  # 0-based
                    while idx >= len(cached.raw_data):
                        cached.raw_data.append([])
                    existing = cached.raw_data[idx]
                    # Extend to 8 columns (A-H) if needed
                    while len(existing) < 8:
                        existing.append('')
                    # Write B-H (indices 1-7)
                    for j, val in enumerate(row_data[:7]):
                        existing[1 + j] = val

            logger.info(f"Wrote {result.get('updatedCells')} cells to {range_str}")
            return True

        except Exception as e:
            logger.error(f"Failed to write expense rows: {e}")
            return False

    def read_income_rows(self, sheet_name):
        """Read income data rows (D to G, between income and savings markers, from cache).

        The income section layout:
          marker row:   ---INCOME---
          marker + 1:   הכנסות header
          marker + 2:   column headers (הערות, כמה, קטגוריה, תאריך)
          marker + 3+:  data rows
          total row:    סה"כ in column G

        Returns:
            List of rows, each [comments, amount, category, date_serial].
            Returns empty list if marker not found.
        """
        cached = self._get_cached(sheet_name)
        if not cached:
            return []
        income_marker = cached.markers.get("income")
        savings_marker = cached.markers.get("savings")
        if not income_marker or not savings_marker:
            return []
        first_data = income_marker + 3  # 1-based
        last_data = savings_marker - 3
        if last_data < first_data:
            return []
        # Convert to 0-based indices
        start_idx = first_data - 1
        end_idx = last_data  # inclusive, so slice to last_data
        rows = cached.raw_data[start_idx:end_idx]
        # Extract columns D:G (indices 3-6)
        return [row[3:7] if len(row) > 3 else [] for row in rows]

    def find_first_empty_income_row(self, sheet_name):
        """Find first empty row in the income section (from cache).

        Scans column G (date) in the income data area.

        Returns:
            1-based row number, or None if markers not found.
        """
        cached = self._get_cached(sheet_name)
        if not cached:
            return None
        income_marker = cached.markers.get("income")
        savings_marker = cached.markers.get("savings")
        if not income_marker or not savings_marker:
            return None
        first_data = income_marker + 3  # 1-based
        last_data = savings_marker - 3
        if last_data < first_data:
            return first_data
        # Scan column G (index 6) in the income data range
        for row_1based in range(first_data, last_data + 1):
            idx = row_1based - 1  # 0-based
            row = cached.raw_data[idx] if idx < len(cached.raw_data) else []
            date_val = row[6] if len(row) > 6 else None
            if not date_val:
                return row_1based
        # All filled
        return last_data + 1

    def write_income_rows(self, sheet_name, start_row, rows):
        """Write income rows to columns D–G.

        Updates cache in-place after a successful API write.

        Args:
            sheet_name: Name of the sheet.
            start_row: 1-based row to start writing.
            rows: List of 4-element lists [D(comments), E(amount), F(category), G(date)].

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
            range_str = f'{sheet_name}!D{start_row}:G{end_row}'

            body = {'values': rows}
            result = self.sheets_service.values().update(
                spreadsheetId=self.spreadsheet_id,
                range=range_str,
                valueInputOption='USER_ENTERED',
                body=body,
            ).execute()

            # Patch cache: write values into columns D-G (indices 3-6)
            cached = self._cache.sheets.get(sheet_name)
            if cached:
                for i, row_data in enumerate(rows):
                    idx = start_row - 1 + i  # 0-based
                    while idx >= len(cached.raw_data):
                        cached.raw_data.append([])
                    existing = cached.raw_data[idx]
                    while len(existing) < 7:
                        existing.append('')
                    for j, val in enumerate(row_data[:4]):
                        existing[3 + j] = val

            logger.info(f"Wrote {result.get('updatedCells')} income cells to {range_str}")
            return True

        except Exception as e:
            logger.error(f"Failed to write income rows: {e}")
            return False

    def read_section_range(self, sheet_name, section, columns, end_section=None):
        """Read a range from a section using cached data.

        Args:
            sheet_name: Name of the sheet.
            section: Section name (e.g. "budget").
            columns: Column range string (e.g. "B:H").
            end_section: Next section name for end boundary. If None,
                         reads 30 rows from marker.

        Returns:
            Tuple of (start_row, list of rows) where start_row is 1-based.
            Returns (None, []) if marker not found.
        """
        cached = self._get_cached(sheet_name)
        if not cached:
            return None, []

        start_row = cached.markers.get(section)
        if start_row is None:
            return None, []

        if end_section:
            end_row = cached.markers.get(end_section)
            if end_row is None:
                end_row = start_row + 30
        else:
            end_row = start_row + 30

        # Parse column range to get column indices
        col_start_str, col_end_str = columns.split(":")
        col_start = ord(col_start_str.upper()) - ord('A')
        col_end = ord(col_end_str.upper()) - ord('A') + 1  # exclusive

        # Extract rows from cache (start_row to end_row-1, both 1-based)
        start_idx = start_row - 1
        end_idx = end_row - 1
        rows = []
        for i in range(start_idx, min(end_idx, len(cached.raw_data))):
            raw_row = cached.raw_data[i]
            # Slice columns
            sliced = raw_row[col_start:col_end] if len(raw_row) > col_start else []
            rows.append(sliced)

        return start_row, rows

    def update_cell(self, sheet_name, cell_ref, value):
        """Write a single value and patch cache.

        Args:
            sheet_name: Name of the sheet.
            cell_ref: Cell reference like "E15".
            value: Value to write.

        Returns:
            True if successful, False otherwise.
        """
        try:
            self.sheets_service.values().update(
                spreadsheetId=self.spreadsheet_id,
                range=f"{sheet_name}!{cell_ref}",
                valueInputOption="USER_ENTERED",
                body={"values": [[value]]},
            ).execute()

            # Patch cache
            cached = self._cache.sheets.get(sheet_name)
            if cached:
                col = ord(cell_ref[0].upper()) - ord('A')
                row_num = int(cell_ref[1:])
                idx = row_num - 1
                while idx >= len(cached.raw_data):
                    cached.raw_data.append([])
                existing = cached.raw_data[idx]
                while len(existing) <= col:
                    existing.append('')
                existing[col] = value

            return True
        except Exception as e:
            logger.error(f"Failed to update cell {cell_ref} in {sheet_name}: {e}")
            return False

    def batch_update_cells(self, sheet_name, updates):
        """Write multiple cell values in a single API call.

        Args:
            sheet_name: Name of the sheet.
            updates: List of (cell_ref, value) tuples,
                     e.g. [("E15", 1234.5), ("E16", "=SUM(...)"), ...].

        Returns:
            True if successful, False otherwise.
        """
        if not updates:
            return True
        try:
            data = [
                {
                    "range": f"{sheet_name}!{ref}",
                    "values": [[val]],
                }
                for ref, val in updates
            ]
            body = {
                "valueInputOption": "USER_ENTERED",
                "data": data,
            }
            result = self.sheets_service.values().batchUpdate(
                spreadsheetId=self.spreadsheet_id,
                body=body,
            ).execute()

            # Patch cache for each update
            cached = self._cache.sheets.get(sheet_name)
            if cached:
                for ref, val in updates:
                    col = ord(ref[0].upper()) - ord('A')
                    row_num = int(ref[1:])
                    idx = row_num - 1
                    while idx >= len(cached.raw_data):
                        cached.raw_data.append([])
                    existing = cached.raw_data[idx]
                    while len(existing) <= col:
                        existing.append('')
                    existing[col] = val

            logger.info(
                f"Batch updated {len(updates)} cells in {sheet_name} "
                f"({result.get('totalUpdatedCells', 0)} total)"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to batch update cells in {sheet_name}: {e}")
            return False

    def find_reconciliation_label_row(self, sheet_name, label):
        """Find a row in the reconciliation section by its label in column F/G.

        The reconciliation section is a series of key-value pairs. Labels
        appear in columns F or G, values in column E. This scans from
        the reconciliation marker downward.

        Args:
            sheet_name: Name of the sheet.
            label: Hebrew label to search for (e.g. 'כסף בעו"ש').

        Returns:
            1-based row number, or None if not found.
        """
        cached = self._get_cached(sheet_name)
        if not cached:
            return None
        recon_marker = cached.markers.get("reconciliation")
        if not recon_marker:
            return None
        # Scan from marker onward (reconciliation is last section)
        start_idx = recon_marker  # 0-based index = marker row (skip marker itself)
        end_idx = min(start_idx + 30, len(cached.raw_data))
        for i in range(start_idx, end_idx):
            row = cached.raw_data[i]
            # Check columns B through G (indices 1-6) for the label
            for col_idx in range(1, min(7, len(row))):
                cell = row[col_idx]
                if isinstance(cell, str) and label in cell:
                    return i + 1  # 1-based
        return None

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
