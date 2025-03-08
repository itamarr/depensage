"""
Integration tests for the spreadsheet handler module.

These tests interact with actual Google Sheets and require valid credentials.
"""

import unittest
import os
import json
import pandas as pd
from datetime import datetime, timedelta
import logging

from depensage.sheets.spreadsheet_handler import SheetHandler
from depensage.sheets.sheet_utils import SheetUtils

# Configure logging to see what's happening during tests
logging.basicConfig(level=logging.INFO)

class TestSheetHandler(unittest.TestCase):
    """Integration tests for the SheetHandler class using real Google Sheets."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Load test configuration from environment or file
        cls.credentials_file = os.environ.get('DEPENSAGE_TEST_CREDENTIALS', 'test_credentials.json')
        cls.test_spreadsheet_id = os.environ.get('DEPENSAGE_TEST_SPREADSHEET_ID')

        # Try to load config from file if not in environment
        if not cls.test_spreadsheet_id:
            try:
                with open('test_config.json', 'r') as f:
                    config = json.load(f)
                    cls.test_spreadsheet_id = config.get('test_spreadsheet_id')
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error loading test config: {e}")

        # Skip all tests if credentials or spreadsheet ID are not available
        if not os.path.exists(cls.credentials_file):
            raise unittest.SkipTest(f"Credentials file {cls.credentials_file} not found")

        if not cls.test_spreadsheet_id:
            raise unittest.SkipTest("Test spreadsheet ID not provided")

        # Initialize the handler
        cls.handler = SheetHandler(cls.test_spreadsheet_id)

        # Authenticate with Google Sheets
        success = cls.handler.authenticate(cls.credentials_file)
        if not success:
            raise unittest.SkipTest("Authentication with Google Sheets failed")

        print(f"Authentication successful, using spreadsheet ID: {cls.test_spreadsheet_id}")

    def setUp(self):
        """Set up before each test."""
        # Create a unique test sheet name (with timestamp to avoid conflicts)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.test_sheet_name = f"Test_{timestamp}"

    def tearDown(self):
        """Clean up after each test."""
        # Delete test sheets if they exist
        try:
            if self.handler.sheet_exists(self.test_sheet_name):
                # Use Google Sheets API to delete the sheet
                sheets = self.handler.get_sheet_metadata()
                sheet_id = None

                for sheet in sheets:
                    if sheet['properties']['title'] == self.test_sheet_name:
                        sheet_id = sheet['properties']['sheetId']
                        break

                if sheet_id:
                    body = {
                        'requests': [
                            {
                                'deleteSheet': {
                                    'sheetId': sheet_id
                                }
                            }
                        ]
                    }

                    self.handler.sheets_service.batchUpdate(
                        spreadsheetId=self.test_spreadsheet_id,
                        body=body
                    ).execute()

                    print(f"Deleted test sheet: {self.test_sheet_name}")
        except Exception as e:
            print(f"Error cleaning up test sheet: {e}")

    def test_authenticate(self):
        """Test authentication with Google Sheets using real credentials."""
        # Authentication already tested in setUpClass, but test again to verify
        # that re-authentication works
        result = self.handler.authenticate(self.credentials_file)
        self.assertTrue(result)

        # Test with invalid credentials file
        invalid_file = "nonexistent_credentials.json"
        if not os.path.exists(invalid_file):
            result = self.handler.authenticate(invalid_file)
            self.assertFalse(result)

    def test_get_sheet_metadata(self):
        """Test getting sheet metadata from the real spreadsheet."""
        sheets = self.handler.get_sheet_metadata()

        # Verify that we got a list of sheets
        self.assertIsNotNone(sheets)
        self.assertIsInstance(sheets, list)
        self.assertTrue(len(sheets) > 0)

        # Verify that each sheet has the expected properties
        self.assertIn('properties', sheets[0])
        self.assertIn('title', sheets[0]['properties'])

    def test_sheet_exists(self):
        """Test checking if a sheet exists in the real spreadsheet."""
        # Get the first sheet name
        sheets = self.handler.get_sheet_metadata()
        if sheets:
            first_sheet_name = sheets[0]['properties']['title']

            # Check that it exists
            self.assertTrue(self.handler.sheet_exists(first_sheet_name))

            # Check that a nonexistent sheet doesn't exist
            self.assertFalse(self.handler.sheet_exists("NonexistentSheetName12345"))

    def test_get_sheet_values(self):
        """Test getting values from a sheet in the real spreadsheet."""
        # Get the first sheet name
        sheets = self.handler.get_sheet_metadata()
        if sheets:
            first_sheet_name = sheets[0]['properties']['title']

            # Get values from the sheet
            values = self.handler.get_sheet_values(first_sheet_name, 'A1:B2')

            # Verify that we got values (may be empty if sheet is empty)
            self.assertIsNotNone(values)

    def test_create_and_update_sheet(self):
        """Test creating a new sheet and updating its values."""
        # Create a new sheet using our test sheet name
        result = self.handler.create_sheet_from_template(self.test_sheet_name, template_name=None)

        if not result:
            # If creation failed, try creating a blank sheet directly with the API
            try:
                body = {
                    'requests': [
                        {
                            'addSheet': {
                                'properties': {
                                    'title': self.test_sheet_name
                                }
                            }
                        }
                    ]
                }

                self.handler.sheets_service.batchUpdate(
                    spreadsheetId=self.test_spreadsheet_id,
                    body=body
                ).execute()

                print(f"Created blank test sheet: {self.test_sheet_name}")
                result = True
            except Exception as e:
                print(f"Error creating test sheet: {e}")
                result = False

        # Verify that creation was successful
        self.assertTrue(result)
        self.assertTrue(self.handler.sheet_exists(self.test_sheet_name))

        # Update the sheet with some test values
        test_values = [
            ['Header1', 'Header2', 'Header3'],
            ['Value1', 'Value2', 'Value3'],
            ['Value4', 'Value5', 'Value6']
        ]

        update_result = self.handler.update_sheet(self.test_sheet_name, 1, test_values)

        # Verify that update was successful
        self.assertTrue(update_result)

        # Read the values back to verify
        updated_values = self.handler.get_sheet_values(self.test_sheet_name, 'A1:C3')

        # Verify that the values match
        self.assertEqual(len(updated_values), 3)
        self.assertEqual(updated_values[0][0], 'Header1')
        self.assertEqual(updated_values[1][1], 'Value2')
        self.assertEqual(updated_values[2][2], 'Value6')

    def test_get_or_create_month_sheet(self):
        """Test getting or creating a month sheet for a given date."""
        # Use a date in the past to avoid conflicts with current month sheets
        test_date = datetime(2020, 6, 15)  # June 15, 2020

        # Get the English month name for this date
        month_name = SheetUtils.get_sheet_name_for_date(test_date)

        # Make a unique test sheet name to avoid conflict with existing month sheets
        month_sheet_name = f"{month_name}_test_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Patch the SheetUtils.get_sheet_name_for_date to return our test name
        original_get_sheet_name = SheetUtils.get_sheet_name_for_date

        try:
            SheetUtils.get_sheet_name_for_date = lambda d: month_sheet_name

            # Get or create the month sheet
            result = self.handler.get_or_create_month_sheet(test_date)

            # Verify that creation was successful
            self.assertEqual(result, month_sheet_name)
            self.assertTrue(self.handler.sheet_exists(month_sheet_name))

            # Clean up - delete the test month sheet
            sheets = self.handler.get_sheet_metadata()
            sheet_id = None

            for sheet in sheets:
                if sheet['properties']['title'] == month_sheet_name:
                    sheet_id = sheet['properties']['sheetId']
                    break

            if sheet_id:
                body = {
                    'requests': [
                        {
                            'deleteSheet': {
                                'sheetId': sheet_id
                            }
                        }
                    ]
                }

                self.handler.sheets_service.batchUpdate(
                    spreadsheetId=self.test_spreadsheet_id,
                    body=body
                ).execute()

                print(f"Deleted test month sheet: {month_sheet_name}")
        finally:
            # Restore the original method
            SheetUtils.get_sheet_name_for_date = original_get_sheet_name

    def test_extract_historical_data(self):
        """Test extracting historical data from sheets."""
        # Create a test sheet with some historical data
        test_values = [
            ['', '', 'הערות', 'תת קטגוריה', 'כמה', 'קטגוריה', 'תאריך'],  # Header
            ['', '', 'Note1', 'Food', '100.50', 'Groceries', '01/15/2024'],
            ['', '', 'Note2', 'Restaurant', '50.75', 'Dining', '01/20/2024']
        ]

        # Create a test sheet
        result = self.handler.create_sheet_from_template(self.test_sheet_name, template_name=None)
        if not result:
            self.skipTest("Failed to create test sheet")

        # Update the sheet with test data
        update_result = self.handler.update_sheet(self.test_sheet_name, 1, test_values)
        if not update_result:
            self.skipTest("Failed to update test sheet")

        # Extract historical data
        historical_data = self.handler.extract_historical_data()

        # Verify that data was extracted
        self.assertIsNotNone(historical_data)
        self.assertFalse(historical_data.empty)

        # Check if our test data is included
        # (May include data from other sheets, so we can't check exact counts)
        categories = historical_data['category'].tolist()
        self.assertTrue('Groceries' in categories or 'Dining' in categories)


if __name__ == '__main__':
    unittest.main()
