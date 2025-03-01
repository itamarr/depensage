"""
Unit tests for the statement parser module.
"""

import unittest
import os
import tempfile
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, mock_open

from depensage.engine.statement_parser import StatementParser


class TestStatementParser(unittest.TestCase):
    """Test cases for the StatementParser class."""

    def setUp(self):
        """Set up the test environment."""
        self.parser = StatementParser()

        # Sample CSV content
        self.sample_csv = (
            "Header row to skip\n"
            "תאריך,עסק,סכום\n"
            "01/02/24,Supermarket,100.50\n"
            "05/02/24,Restaurant,50.75\n"
            "10/02/24,Gas Station,30.25\n"
        )

        # Create a temporary CSV file
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        self.temp_file.write(self.sample_csv.encode('utf-8-sig'))
        self.temp_file.close()

    def tearDown(self):
        """Clean up temporary files."""
        os.unlink(self.temp_file.name)

    def test_parse_statement(self):
        """Test parsing a credit card statement."""
        # Test with the temporary file
        with patch('pandas.read_csv') as mock_read_csv:
            # Create a mock dataframe
            mock_df = pd.DataFrame({
                'תאריך': ['01/02/24', '05/02/24', '10/02/24'],
                'עסק': ['Supermarket', 'Restaurant', 'Gas Station'],
                'סכום': ['100.50', '50.75', '30.25']
            })
            mock_read_csv.return_value = mock_df

            # Mock the date conversion
            with patch.object(pd.Series, 'astype') as mock_astype:
                mock_astype.return_value = mock_df['סכום']

                with patch.object(pd.Series, 'str') as mock_str:
                    mock_str.return_value.replace.return_value = mock_df['סכום']

                    with patch.object(pd.Series, 'dt') as mock_dt:
                        mock_dt.dayofweek = pd.Series([0, 1, 2])

                        # Parse the statement
                        result = self.parser.parse_statement(self.temp_file.name)

                        # Check the result
                        self.assertIsNotNone(result)
                        mock_read_csv.assert_called_once()

        # Test error handling
        with patch('pandas.read_csv', side_effect=Exception("CSV error")):
            result = self.parser.parse_statement('nonexistent.csv')
            self.assertIsNone(result)

    def test_merge_statements(self):
        """Test merging multiple statements."""
        # Create sample dataframes
        df1 = pd.DataFrame({
            'date': [datetime(2024, 2, 1), datetime(2024, 2, 5)],
            'business_name': ['Supermarket', 'Restaurant'],
            'amount': [100.50, 50.75]
        })

        df2 = pd.DataFrame({
            'date': [datetime(2024, 2, 3), datetime(2024, 2, 8)],
            'business_name': ['Gas Station', 'Pharmacy'],
            'amount': [30.25, 25.00]
        })

        # Test merging
        result = self.parser.merge_statements(df1, df2)

        # Check the result
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 4)  # 2 + 2 = 4 transactions
        self.assertEqual(result.iloc[0]['business_name'], 'Supermarket')  # First by date
        self.assertEqual(result.iloc[-1]['business_name'], 'Pharmacy')  # Last by date

        # Test with empty dataframes
        result = self.parser.merge_statements(pd.DataFrame(), pd.DataFrame())
        self.assertIsNone(result)

        # Test with None values
        result = self.parser.merge_statements(df1, None)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)  # Only df1 transactions

        # Test error handling
        with patch('pandas.concat', side_effect=Exception("Concat error")):
            result = self.parser.merge_statements(df1, df2)
            self.assertIsNone(result)

    def test_filter_new_transactions(self):
        """Test filtering new transactions."""
        # Create sample dataframe
        df = pd.DataFrame({
            'date': [
                datetime(2024, 2, 1),
                datetime(2024, 2, 5),
                datetime(2024, 2, 10),
                datetime(2024, 2, 15)
            ],
            'business_name': ['Supermarket', 'Restaurant', 'Gas Station', 'Pharmacy'],
            'amount': [100.50, 50.75, 30.25, 25.00]
        })

        # Test filtering with cutoff date
        cutoff_date = datetime(2024, 2, 8)
        result = self.parser.filter_new_transactions(df, cutoff_date)

        # Check the result
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)  # 2 transactions after cutoff
        self.assertEqual(result.iloc[0]['business_name'], 'Gas Station')
        self.assertEqual(result.iloc[1]['business_name'], 'Pharmacy')

        # Test with empty dataframe
        result = self.parser.filter_new_transactions(pd.DataFrame(), cutoff_date)
        self.assertIsNone(result)

        # Test with None dataframe
        result = self.parser.filter_new_transactions(None, cutoff_date)
        self.assertIsNone(result)

    def test_format_transactions_for_sheet(self):
        """Test formatting transactions for Google Sheet."""
        # Create sample classified dataframe
        df = pd.DataFrame({
            'date': [datetime(2024, 2, 1), datetime(2024, 2, 5)],
            'business_name': ['Supermarket', 'Restaurant'],
            'amount': [100.50, 50.75],
            'predicted_category': ['Groceries', 'Dining'],
            'predicted_subcategory': ['Food', 'Restaurant']
        })

        # Test formatting
        result = self.parser.format_transactions_for_sheet(df)

        # Check the result
        self.assertEqual(len(result), 2)  # 2 formatted transactions

        # Check the structure of the first transaction
        self.assertEqual(len(result[0]), 7)  # 7 columns
        self.assertEqual(result[0][3], 'Food')  # Subcategory
        self.assertEqual(result[0][4], '100.50')  # Amount
        self.assertEqual(result[0][5], 'Groceries')  # Category
        self.assertEqual(result[0][6], '02/01/2024')  # Date in MM/DD/YYYY format

        # Test with empty dataframe
        result = self.parser.format_transactions_for_sheet(pd.DataFrame())
        self.assertEqual(result, [])

        # Test with None dataframe
        result = self.parser.format_transactions_for_sheet(None)
        self.assertEqual(result, [])


if __name__ == '__main__':
    unittest.main()
