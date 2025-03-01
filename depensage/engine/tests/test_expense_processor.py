"""
Unit tests for the expense processor module.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime

from depensage.engine.expense_processor import ExpenseProcessor


class TestExpenseProcessor(unittest.TestCase):
    """Test cases for the ExpenseProcessor class."""

    def setUp(self):
        """Set up the test environment."""
        # Mock dependencies
        with patch('depensage.classifier.neural_classifier.ExpenseNeuralClassifier') as mock_classifier, \
                patch('depensage.sheets.spreadsheet_handler.SheetHandler') as mock_sheet_handler, \
                patch('depensage.engine.statement_parser.StatementParser') as mock_parser:
            # Create mocks
            self.mock_classifier = mock_classifier.return_value
            self.mock_sheet_handler = mock_sheet_handler.return_value
            self.mock_parser = mock_parser.return_value

            # Create processor
            self.processor = ExpenseProcessor('test_spreadsheet_id', 'test_credentials.json')

    def test_authenticate(self):
        """Test authentication with Google Sheets."""
        # Test successful authentication
        self.mock_sheet_handler.authenticate.return_value = True

        result = self.processor.authenticate('new_credentials.json')

        # Check the result
        self.assertTrue(result)
        self.mock_sheet_handler.authenticate.assert_called_once_with('new_credentials.json')
        self.assertEqual(self.processor.credentials_file, 'new_credentials.json')

        # Test authentication failure
        self.mock_sheet_handler.authenticate.return_value = False

        result = self.processor.authenticate()

        # Check the result
        self.assertFalse(result)

        # Test with no credentials
        result = self.processor.authenticate(None)

        # Check the result
        self.assertFalse(result)

    def test_train_classifier(self):
        """Test training the classifier."""
        # Mock historical data
        self.mock_sheet_handler.extract_historical_data.return_value = pd.DataFrame({
            'date': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            'business_name': ['Supermarket', 'Restaurant'],
            'amount': [100.50, 50.75],
            'category': ['Groceries', 'Dining'],
            'subcategory': ['Food', 'Restaurant']
        })

        # Mock training
        mock_history = {'accuracy': [0.7, 0.8, 0.9]}
        self.mock_classifier.train.return_value = mock_history

        # Test training
        result = self.processor.train_classifier()

        # Check the result
        self.assertEqual(result, mock_history)
        self.mock_sheet_handler.extract_historical_data.assert_called_once()
        self.mock_classifier.train.assert_called_once()

        # Test with additional data
        additional_data = pd.DataFrame({
            'date': [datetime(2024, 1, 3)],
            'business_name': ['Gas Station'],
            'amount': [30.25],
            'category': ['Transportation'],
            'subcategory': ['Fuel']
        })

        self.mock_sheet_handler.extract_historical_data.reset_mock()
        self.mock_classifier.train.reset_mock()

        result = self.processor.train_classifier(additional_data)

        # Check the result - should use both data sources
        self.assertEqual(result, mock_history)
        self.mock_sheet_handler.extract_historical_data.assert_called_once()
        self.mock_classifier.train.assert_called_once()

        # Test with no data
        self.mock_sheet_handler.extract_historical_data.return_value = pd.DataFrame()

        with patch('pandas.concat') as mock_concat:
            # Empty DataFrame for additional data
            result = self.processor.train_classifier(pd.DataFrame())

            # Check the result - should fail with no data
            self.assertIsNone(result)
            mock_concat.assert_not_called()

        # Test error handling
        self.mock_sheet_handler.extract_historical_data.side_effect = Exception("API error")

        result = self.processor.train_classifier()

        # Check the result - should fail
        self.assertIsNone(result)

    def test_process_statement(self):
        """Test processing credit card statements."""
        # Mock parsed statement data
        primary_data = pd.DataFrame({
            'date': [datetime(2024, 2, 1), datetime(2024, 2, 5)],
            'business_name': ['Supermarket', 'Restaurant'],
            'amount': [100.50, 50.75]
        })

        secondary_data = pd.DataFrame({
            'date': [datetime(2024, 2, 3), datetime(2024, 2, 8)],
            'business_name': ['Gas Station', 'Pharmacy'],
            'amount': [30.25, 25.00]
        })

        merged_data = pd.concat([primary_data, secondary_data], ignore_index=True)

        # Mock parsing and merging
        self.mock_parser.parse_statement.side_effect = [primary_data, secondary_data]
        self.mock_parser.merge_statements.return_value = merged_data

        # Mock classification
        classified_data = merged_data.copy()
        classified_data['predicted_category'] = ['Groceries', 'Dining', 'Transportation', 'Health']
        classified_data['predicted_subcategory'] = ['Food', 'Restaurant', 'Fuel', 'Medicine']
        self.mock_classifier.predict.return_value = classified_data

        # Mock month sheet handling
        self.mock_sheet_handler.get_or_create_month_sheet.return_value = 'February'
        self.mock_sheet_handler.get_last_updated_date.return_value = datetime(2024, 1, 31)

        # Mock transaction filtering
        self.mock_parser.filter_new_transactions.return_value = classified_data

        # Mock formatting
        formatted_data = [['', '', '', 'Food', '100.50', 'Groceries', '02/01/2024']]
        self.mock_parser.format_transactions_for_sheet.return_value = formatted_data

        # Mock sheet values and update
        self.mock_sheet_handler.get_sheet_values.return_value = [['Header']]
        self.mock_sheet_handler.update_sheet.return_value = True

        # Test processing
        result = self.processor.process_statement('primary.csv', 'secondary.csv')

        # Check the result
        self.assertTrue(result)
        self.mock_parser.parse_statement.assert_any_call('primary.csv')
        self.mock_parser.parse_statement.assert_any_call('secondary.csv')
        self.mock_parser.merge_statements.assert_called_once()
        self.mock_classifier.predict.assert_called_once()
        self.mock_sheet_handler.get_or_create_month_sheet.assert_called()
        self.mock_sheet_handler.update_sheet.assert_called()

        # Test with only primary statement
        self.mock_parser.parse_statement.reset_mock()
        self.mock_parser.merge_statements.reset_mock()
        self.mock_classifier.predict.reset_mock()

        self.mock_parser.parse_statement.return_value = primary_data
        self.mock_parser.merge_statements.return_value = primary_data

        result = self.processor.process_statement('primary.csv')

        # Check the result
        self.assertTrue(result)
        self.mock_parser.parse_statement.assert_called_once_with('primary.csv')
        self.mock_parser.merge_statements.assert_called_once()
        self.mock_classifier.predict.assert_called_once()

        # Test with no transactions
        self.mock_parser.parse_statement.reset_mock()
        self.mock_parser.merge_statements.reset_mock()
        self.mock_classifier.predict.reset_mock()

        self.mock_parser.merge_statements.return_value = None

        result = self.processor.process_statement('primary.csv')

        # Check the result - should succeed with no work
        self.assertTrue(result)
        self.mock_classifier.predict.assert_not_called()

        # Test with sheet creation failure
        self.mock_parser.parse_statement.reset_mock()
        self.mock_parser.merge_statements.reset_mock()

        self.mock_parser.merge_statements.return_value = primary_data
        self.mock_sheet_handler.get_or_create_month_sheet.return_value = None

        result = self.processor.process_statement('primary.csv')

        # Check the result - should indicate failure
        self.assertFalse(result)

        # Test with sheet update failure
        self.mock_sheet_handler.get_or_create_month_sheet.return_value = 'February'
        self.mock_sheet_handler.update_sheet.return_value = False

        result = self.processor.process_statement('primary.csv')

        # Check the result - should indicate failure
        self.assertFalse(result)

        # Test error handling
        self.mock_parser.parse_statement.side_effect = Exception("Parser error")

        result = self.processor.process_statement('primary.csv')

        # Check the result - should fail
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
