#!/usr/bin/env python3
"""
DepenSage - Automated Expense Tracking

Main entry point for the DepenSage application.
"""

import argparse
import logging
import sys
import os
from datetime import datetime

from depensage.config.settings import Settings
from depensage.engine.expense_processor import ExpenseProcessor


def setup_logging(log_level):
    """
    Configure logging.

    Args:
        log_level: Logging level string.
    """
    # Map string log levels to logging constants
    log_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }

    level = log_levels.get(log_level, logging.INFO)

    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('depensage.log')
        ]
    )


def configure(args):
    """
    Configure DepenSage settings.

    Args:
        args: Command line arguments.

    Returns:
        0 if successful, 1 otherwise.
    """
    settings = Settings()

    # Set provided values
    updates = {}

    if args.spreadsheet_id:
        updates['spreadsheet_id'] = args.spreadsheet_id

    if args.credentials_file:
        updates['credentials_file'] = args.credentials_file

    if args.log_level:
        updates['log_level'] = args.log_level

    # Save settings
    if updates:
        success = settings.set_multiple(updates)
        if success:
            print("Settings updated successfully.")
            return 0
        else:
            print("Failed to update settings.")
            return 1
    else:
        # Show current settings
        print("Current DepenSage settings:")
        for key, value in settings.settings.items():
            print(f"  {key}: {value}")
        return 0


def process(args):
    """
    Process credit card statements.

    Args:
        args: Command line arguments.

    Returns:
        0 if successful, 1 otherwise.
    """
    settings = Settings()

    # Load required settings
    spreadsheet_id = args.spreadsheet_id or settings.get('spreadsheet_id')
    credentials_file = args.credentials_file or settings.get('credentials_file')

    if not spreadsheet_id:
        print("Error: No spreadsheet ID provided.")
        return 1

    if not credentials_file:
        print("Error: No credentials file provided.")
        return 1

    if not args.primary_statement:
        print("Error: No primary statement file provided.")
        return 1

    # Check file existence
    if not os.path.exists(args.primary_statement):
        print(f"Error: Primary statement file {args.primary_statement} does not exist.")
        return 1

    if args.secondary_statement and not os.path.exists(args.secondary_statement):
        print(f"Error: Secondary statement file {args.secondary_statement} does not exist.")
        return 1

    try:
        # Create the processor
        processor = ExpenseProcessor(
            spreadsheet_id=spreadsheet_id,
            credentials_file=credentials_file,
        )

        # Process statements
        print("Processing credit card statements...")
        success = processor.process_statement(
            args.primary_statement,
            args.secondary_statement
        )

        if success:
            print("Statements processed successfully!")
            return 0
        else:
            print("Statement processing failed. Check the logs for details.")
            return 1
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1


def main():
    """
    Main entry point.
    """
    # Create main parser
    parser = argparse.ArgumentParser(
        description='DepenSage - Automated Expense Tracking'
    )

    # Set up subparsers
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Configure command
    configure_parser = subparsers.add_parser('configure', help='Configure DepenSage settings')
    configure_parser.add_argument('--spreadsheet-id', help='Google Spreadsheet ID')
    configure_parser.add_argument('--credentials-file', help='Path to Google API credentials file')
    configure_parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                                  help='Logging level')

    # Process command
    process_parser = subparsers.add_parser('process', help='Process credit card statements')
    process_parser.add_argument('--spreadsheet-id', help='Google Spreadsheet ID (overrides config)')
    process_parser.add_argument('--credentials-file', help='Path to Google API credentials file (overrides config)')
    process_parser.add_argument('primary_statement', help='Primary credit card statement CSV file')
    process_parser.add_argument('secondary_statement', nargs='?', help='Secondary credit card statement CSV file')

    # Parse arguments
    args = parser.parse_args()

    # Load settings
    settings = Settings()

    # Configure logging
    setup_logging(settings.get('log_level', 'INFO'))

    # Run appropriate command
    if args.command == 'configure':
        return configure(args)
    elif args.command == 'process':
        return process(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
