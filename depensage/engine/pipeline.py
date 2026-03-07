"""
Core automated expense pipeline.

Parses CC statements, filters pending transactions, classifies merchants,
deduplicates against existing sheet data, and writes to the correct
monthly Google Sheets.
"""

import logging
from dataclasses import dataclass, field

import pandas as pd

from depensage.engine.statement_parser import StatementParser
from depensage.engine.dedup import deduplicate
from depensage.engine.formatter import format_for_sheet
from depensage.sheets.spreadsheet_handler import SheetHandler, EXPENSE_END_MARKER
from depensage.classifier.lookup import LookupClassifier

logger = logging.getLogger(__name__)


@dataclass
class MonthResult:
    month: str
    year: int
    written: int
    duplicates: int


@dataclass
class PipelineResult:
    total_parsed: int
    pending_skipped: int
    classified: int
    unclassified: int
    months: list[MonthResult] = field(default_factory=list)
    unclassified_merchants: list[str] = field(default_factory=list)


def run_pipeline(statement_paths, handlers, classifier, year=None):
    """Run the full expense processing pipeline.

    Args:
        statement_paths: List of file paths to CC statement files.
        handlers: Dict mapping year (int) to authenticated SheetHandler,
                  or a single SheetHandler (used for all years).
        classifier: LookupClassifier instance.
        year: Optional year filter (int). If set, only transactions
              from this year are processed.

    Returns:
        PipelineResult with processing statistics.

    Raises:
        ValueError: If marker not found or no handler for a year.
    """
    parser = StatementParser()

    # Normalize handlers: dict of {year: handler} or single handler for all years
    if isinstance(handlers, dict):
        handler_dict = handlers
        single_handler = None
    else:
        handler_dict = None
        single_handler = handlers

    def get_handler(tx_year):
        if single_handler is not None:
            return single_handler
        if tx_year not in handler_dict:
            raise ValueError(
                f"No spreadsheet handler configured for year {tx_year}. "
                f"Available: {sorted(handler_dict.keys())}"
            )
        return handler_dict[tx_year]

    # 1. Parse all files and merge
    dfs = []
    for path in statement_paths:
        df = parser.parse_statement(path)
        if df is not None and not df.empty:
            dfs.append(df)

    if not dfs:
        return PipelineResult(total_parsed=0, pending_skipped=0,
                              classified=0, unclassified=0)

    all_transactions = pd.concat(dfs, ignore_index=True).sort_values("date")
    total_parsed = len(all_transactions)

    # 2. Filter pending
    charged = StatementParser.filter_pending(all_transactions)
    pending_skipped = total_parsed - len(charged)

    if charged.empty:
        return PipelineResult(total_parsed=total_parsed,
                              pending_skipped=pending_skipped,
                              classified=0, unclassified=0)

    # 3. Apply year filter
    if year is not None:
        charged = charged[charged["date"].dt.year == year].reset_index(drop=True)
        if charged.empty:
            return PipelineResult(total_parsed=total_parsed,
                                  pending_skipped=pending_skipped,
                                  classified=0, unclassified=0)

    # 4. Classify
    result = classifier.classify(charged)
    classified_count = len(result.classified)
    unclassified_count = len(result.unclassified)
    unclassified_merchants = (
        result.unclassified["business_name"].unique().tolist()
        if not result.unclassified.empty else []
    )

    # 5. Recombine: classified + unclassified (with empty category/subcategory)
    if not result.unclassified.empty:
        unclassified_with_cols = result.unclassified.copy()
        unclassified_with_cols["category"] = ""
        unclassified_with_cols["subcategory"] = ""
        combined = pd.concat(
            [result.classified, unclassified_with_cols], ignore_index=True
        ).sort_values("date")
    else:
        combined = result.classified

    # 6. Group by year-month
    combined["_year_month"] = combined["date"].dt.to_period("M")
    month_results = []

    for period, group in combined.groupby("_year_month"):
        sample_date = group["date"].iloc[0]
        tx_year = sample_date.year
        handler = get_handler(tx_year)

        sheet_name = handler.get_or_create_month_sheet(sample_date)

        if not sheet_name:
            logger.error(f"Failed to get/create sheet for {period}")
            continue

        # Find marker
        marker_row = handler.find_expense_end_row(sheet_name)
        if marker_row is None:
            raise ValueError(
                f"No {EXPENSE_END_MARKER} marker found in column B of "
                f"sheet '{sheet_name}'. Run the marker migration script first."
            )

        # Read existing rows for dedup
        existing_rows = handler.read_expense_rows(sheet_name)

        # Deduplicate
        month_txns = group.drop(columns=["_year_month"])
        new_txns = deduplicate(month_txns, existing_rows)
        dup_count = len(month_txns) - len(new_txns)

        if new_txns.empty:
            month_results.append(MonthResult(
                month=sheet_name, year=tx_year, written=0, duplicates=dup_count
            ))
            continue

        # Format for sheet
        formatted = format_for_sheet(new_txns)

        # Find insertion point
        first_empty = handler.find_first_empty_expense_row(sheet_name)
        rows_needed = len(formatted)
        available = marker_row - first_empty

        if available < rows_needed:
            insert_count = rows_needed - available
            handler.insert_rows(sheet_name, marker_row, insert_count)

        # Write
        handler.write_expense_rows(sheet_name, first_empty, formatted)

        month_results.append(MonthResult(
            month=sheet_name, year=tx_year,
            written=len(formatted), duplicates=dup_count
        ))

    return PipelineResult(
        total_parsed=total_parsed,
        pending_skipped=pending_skipped,
        classified=classified_count,
        unclassified=unclassified_count,
        months=month_results,
        unclassified_merchants=unclassified_merchants,
    )
