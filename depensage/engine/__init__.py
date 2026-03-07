"""
DepenSage Engine Subpackage

This subpackage handles the main processing engine for expense tracking.
"""

from depensage.engine.statement_parser import StatementParser
from depensage.engine.pipeline import run_pipeline

__all__ = ['StatementParser', 'run_pipeline']
