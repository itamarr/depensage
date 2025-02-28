"""
Unit tests for the spreadsheet handler module.
"""

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime
import pandas as pd

from depensage.sheets.spreadsheet_handler import SheetHandler


class TestSheetHandler(unittest.TestCase):
    """Test cases for the SheetHandler class."""

    def setUp(self):
        """Set up the test environment."""
        self.handler = SheetHandler('test_spreadsheet_id')

        # Mock the sheets_service
        self.handler.sheets_service = MagicMock()

    def test_authenticate(self):
        """Test authentication with Google Sheets."""
        # Mock the service_account module
        with patch('google.oauth2.service_account.Credentials') as mock_creds, \
                patch('googleapiclient.discovery.build') as mock_build:
            # Set up return values
            mock_creds.from_service_account_file.return_value = MagicMock()
            mock_service = MagicMock()
            mock_service.spreadsheets.return_value = MagicMock()
            mock_build.return_value = mock_service

            # Test successful authentication
            result = self.handler.authenticate('fake_credentials.json')

            # Check that authentication was successful
            self.assertTrue(result)
            mock_creds.from_service_account_file.assert_called_once()
            mock_build.assert_called_once()

            # Test authentication failure
            mock_creds.from_service_account_file.side_effect = Exception("Auth error")

            result = self.handler.authenticate('fake_credentials.json')

            # Check that authentication failed
            self.assertFalse(result)

    def test_get_sheet_metadata(self):
        """Test getting sheet metadata."""
        # Mock the API response
        mock_response = {
            'sheets': [
                {'properties': {'title': 'Sheet1', 'sheetId': 123}},
                {'properties': {'title': 'Sheet2', 'sheetId': 456}}
            ]
        }
        self.handler.sheets_service.get.return_value.execute.return_value = mock_response

        # Get metadata
        result = self.handler.get_sheet_metadata()

        # Check the result
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['properties']['title'], 'Sheet1')

        # Test error handling
        self.handler.sheets_service.get.side_effect = Exception("API error")
        result = self.handler.get_sheet_metadata()
        self.assertIsNone(result)

    def test_sheet_exists(self):
        """Test checking if a sheet exists."""
        # Mock get_sheet_metadata
        with patch.object(self.handler, 'get_sheet_metadata') as mock_get_metadata:
            mock_get_metadata.return_value = [
                {'properties': {'title': 'Sheet1'}},
                {'properties': {'title': 'Sheet2'}}
            ]

            # Test existing sheet
            self.assertTrue(self.handler.sheet_exists('Sheet1'))

            # Test non-existing sheet
            self.assertFalse(self.handler.sheet_exists('Sheet3'))

            # Test when metadata is None
            mock_get_metadata.return_value = None
            self.assertFalse(self.handler.sheet_exists('Sheet1'))

    def test_get_sheet_values(self):
        """Test getting values from a sheet."""
        # Mock the API response
        mock_response = {
            'values': [
                ['Header1', 'Header2'],
                ['Value1', 'Value2']
            ]
        }
        self.handler.sheets_service.values().get.return_value.execute.return_value = mock_response

        # Get values
        result = self.handler.get_sheet_values('Sheet1', 'A1:B2')

        # Check the result
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], 'Header1')

        # Test empty values
        mock_response = {}  # No 'values' key
        self.handler.sheets_service.values().get.return_value.execute.return_value = mock_response

        result = self.handler.get_sheet_values('Sheet1', 'A1:B2')
        self.assertEqual(result, [])

        # Test error handling
        self.handler.sheets_service.values().get.side_effect = Exception("API error")
        result = self.handler.get_sheet_values('Sheet1', 'A1:B2')
        self.assertIsNone(result)

    def test_get_last_updated_date(self):
        """Test getting the last updated date from a sheet."""
        # Mock get_sheet_values
        with patch.object(self.handler, 'get_sheet_values') as mock_get_values:
            # Test with valid date
            mock_get_values.return_value = [
                ['תאריך'],  # Header
                ['01/15/2024'],  # Valid date
                ['01/20/2024']  # Latest date
            ]

            result = self.handler.get_last_updated_date('Sheet1')

            # Check the result - should be the last date
            self.assertEqual(result, datetime(2024, 1, 20))

            # Test with no values
            mock_get_values.return_value = []

            result = self.handler.get_last_updated_date('Sheet1')

            # Check the result - should be a very old date
            self.assertEqual(result, datetime(1900, 1, 1))

            # Test with invalid date
            mock_get_values.return_value = [
                ['תאריך'],
                ['invalid-date']
            ]

            result = self.handler.get_last_updated_date('Sheet1')

            # Check the result - should be a very old date
            self.assertEqual(result, datetime(1900, 1, 1))

    def test_create_sheet_from_template(self):
        """Test creating a sheet from a template."""
        # Mock necessary methods
        with patch.object(self.handler, 'sheet_exists') as mock_exists, \
                patch.object(self.handler, 'get_sheet_metadata') as mock_get_metadata:
            # Test when sheet already exists
            mock_exists.return_value = True

            result = self.handler.create_sheet_from_template('NewSheet')

            # Check the result - should be True without API call
            self.assertTrue(result)
            self.handler.sheets_service.batchUpdate.assert_not_called()

            # Test when sheet doesn't exist
            mock_exists.return_value = False
            mock_get_metadata.return_value = [
                {'properties': {'title': 'Month Template', 'sheetId': 123}}
            ]

            result = self.handler.create_sheet_from_template('NewSheet')

            # Check the result - should be True with API call
            self.assertTrue(result)
            self.handler.sheets_service.batchUpdate.assert_called_once()

            # Test when template doesn't exist
            mock_get_metadata.return_value = [
                {'properties': {'title': 'OtherSheet', 'sheetId': 456}}
            ]
            self.handler.sheets_service.batchUpdate.reset_mock()

            result = self.handler.create_sheet_from_template('NewSheet')

            # Check the result - should be False without API call
            self.assertFalse(result)
            self.handler.sheets_service.batchUpdate.assert_not_called()

    def test_update_sheet(self):
        """Test updating values in a sheet."""
        # Test with values
        values = [['Value1', 'Value2'], ['Value3', 'Value4']]
        mock_response = {'updatedCells': 4}
        self.handler.sheets_service.values().update.return_value.execute.return_value = mock_response

        result = self.handler.update_sheet('Sheet1', 1, values)

        # Check the result - should be True
        self.assertTrue(result)
        self.handler.sheets_service.values().update.assert_called_once()

        # Test with no values
        self.handler.sheets_service.values().update.reset_mock()

        result = self.handler.update_sheet('Sheet1', 1, [])

        # Check the result - should be True without API call
        self.assertTrue(result)
        self.handler.sheets_service.values().update.assert_not_called()

        # Test error handling
        self.handler.sheets_service.values().update.side_effect = Exception("API error")

        result = self.handler.update_sheet('Sheet1', 1, values)

        # Check the result - should be False
        self.assertFalse(result)

    def test_get_or_create_month_sheet(self):
        """Test getting or creating a month sheet."""
        # Mock methods
        with patch.object(self.handler, 'sheet_exists') as mock_exists, \
                patch.object(self.handler, 'create_sheet_from_template') as mock_create:
            # Test when sheet exists
            mock_exists.return_value = True

            result = self.handler.get_or_create_month_sheet(datetime(2024, 1, 15))

            # Check the result - should be the Hebrew month name
            self.assertEqual(result, 'ינואר')
            mock_create.assert_not_called()

            # Test when sheet doesn't exist but creation succeeds
            mock_exists.return_value = False
            mock_create.return_value = True

            result = self.handler.get_or_create_month_sheet(datetime(2024, 2, 15))

            # Check the result - should be the Hebrew month name
            self.assertEqual(result, 'פברואר')
            mock_create.assert_called_once_with('פברואר')

            # Test when sheet doesn't exist and creation fails
            mock_create.reset_mock()
            mock_create.return_value = False

            result = self.handler.get_or_create_month_sheet(datetime(2024, 3, 15))

            # Check the result - should be None
            self.assertIsNone(result)
            mock_create.assert_called_once_with('מרץ')

    def test_extract_historical_data(self):
        """Test extracting historical data from sheets."""
        # Mock methods
        with patch.object(self.handler, 'get_sheet_metadata') as mock_get_metadata, \
                patch.object(self.handler, 'get_sheet_values') as mock_get_values:

            # Test with valid data
            mock_get_metadata.return_value = [
                {'properties': {'title': 'ינואר'}},  # January
                {'properties': {'title': 'פברואר'}},  # February
                {'properties': {'title': 'Categories'}},  # Non-month sheet
                {'properties': {'title': 'Month Template'}}  # Template
            ]

            # Return different values for different sheets
            def mock_get_values_side_effect(sheet_name, range_name):
                if sheet_name == 'ינואר':
                    return [
                        ['', '', 'הערות', 'תת קטגוריה', 'כמה', 'קטגוריה', 'תאריך'],  # Header
                        ['', '', 'Note1', 'Food', '100', 'Groceries', '01/05/2024'],
                        ['', '', 'Note2', 'Restaurant', '50', 'Dining', '01/10/2024']
                    ]
                elif sheet_name == 'פברואר':
                    return [
                        ['', '', 'הערות', 'תת קטגוריה', 'כמה', 'קטגוריה', 'תאריך'],  # Header
                        ['', '', 'Note3', 'Fuel', '30', 'Transportation', '02/05/2024']
                    ]
                else:
                    return []

            mock_get_values.side_effect = mock_get_values_side_effect

            result = self.handler.extract_historical_data()

            # Check the result - should have 3 transactions
            self.assertEqual(len(result), 3)
            self.assertEqual(result['category'].tolist(), ['Groceries', 'Dining', 'Transportation'])

            # Test with no sheets
            mock_get_metadata.return_value = []

            result = self.handler.extract_historical_data()

            # Check the result - should be empty DataFrame
            self.assertTrue(result.empty)

            # Test with sheets but no data
            mock_get_metadata.return_value = [
                {'properties': {'title': 'ינואר'}}
            ]
            mock_get_values.side_effect = lambda sheet_name, range_name: []

            result = self.handler.extract_historical_data()

            # Check the result - should be empty DataFrame
            self.assertTrue(result.empty)

            # Test error handling
            mock_get_metadata.side_effect = Exception("API error")

            result = self.handler.extract_historical_data()

            # Check the result - should be empty DataFrame
            self.assertTrue(result.empty)


if __name__ == '__main__':
    unittest.main()
