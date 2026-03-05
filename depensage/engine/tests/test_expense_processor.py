"""
Unit tests for the expense processor module.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime

from depensage.engine.expense_processor import ExpenseProcessor

MODULE = 'depensage.engine.expense_processor'


class TestExpenseProcessor(unittest.TestCase):

    def setUp(self):
        with patch(f'{MODULE}.SheetHandler') as mock_sh_cls, \
                patch(f'{MODULE}.StatementParser') as mock_parser_cls:
            self.mock_sheet_handler = mock_sh_cls.return_value
            self.mock_parser = mock_parser_cls.return_value
            self.mock_sheet_handler.authenticate.return_value = True
            self.processor = ExpenseProcessor('test_id', 'test_creds.json')

    def test_authenticate(self):
        self.mock_sheet_handler.authenticate.return_value = True
        result = self.processor.authenticate('new_creds.json')
        self.assertTrue(result)
        self.assertEqual(self.processor.credentials_file, 'new_creds.json')

        self.mock_sheet_handler.authenticate.return_value = False
        result = self.processor.authenticate()
        self.assertFalse(result)

    def test_authenticate_no_credentials(self):
        self.processor.credentials_file = None
        result = self.processor.authenticate(None)
        self.assertFalse(result)

    def test_process_statement(self):
        primary_data = pd.DataFrame({
            'date': pd.to_datetime(['2024-02-01', '2024-02-05']),
            'business_name': ['Supermarket', 'Restaurant'],
            'amount': [100.50, 50.75]
        })
        merged_data = primary_data.copy()

        self.mock_parser.parse_statement.return_value = primary_data
        self.mock_parser.merge_statements.return_value = merged_data
        self.mock_sheet_handler.get_or_create_month_sheet.return_value = 'February'
        self.mock_sheet_handler.get_last_updated_date.return_value = datetime(2024, 1, 31)
        self.mock_parser.filter_new_transactions.return_value = merged_data
        self.mock_parser.format_transactions_for_sheet.return_value = [['row']]
        self.mock_sheet_handler.get_sheet_values.return_value = [['Header']]
        self.mock_sheet_handler.update_sheet.return_value = True

        result = self.processor.process_statement('primary.csv')
        self.assertTrue(result)
        self.mock_parser.parse_statement.assert_called_once_with('primary.csv')
        self.mock_sheet_handler.update_sheet.assert_called()

    def test_process_statement_no_transactions(self):
        self.mock_parser.parse_statement.return_value = pd.DataFrame()
        self.mock_parser.merge_statements.return_value = None
        result = self.processor.process_statement('primary.csv')
        self.assertTrue(result)

    def test_process_statement_sheet_creation_failure(self):
        data = pd.DataFrame({
            'date': pd.to_datetime(['2024-02-01']),
            'business_name': ['Supermarket'],
            'amount': [100.50]
        })
        self.mock_parser.parse_statement.return_value = data
        self.mock_parser.merge_statements.return_value = data
        self.mock_sheet_handler.get_or_create_month_sheet.return_value = None
        result = self.processor.process_statement('primary.csv')
        self.assertFalse(result)

    def test_process_statement_error(self):
        self.mock_parser.parse_statement.side_effect = Exception("Parser error")
        result = self.processor.process_statement('primary.csv')
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
