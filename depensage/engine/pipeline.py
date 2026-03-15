"""
Core automated expense pipeline.

Parses CC and bank statements, filters in-process transactions, classifies,
deduplicates against existing sheet data, and stages writes for the correct
monthly Google Sheets. Returns a StagedPipelineResult that the caller can
inspect, export, and commit.
"""

import logging
from dataclasses import dataclass, field

import pandas as pd

from depensage.engine.statement_parser import StatementParser
from depensage.engine.bank_parser import (
    detect_bank_transcript, parse_bank_transcript, BankParseResult, CCLumpSum,
)
from depensage.engine.dedup import deduplicate, deduplicate_income
from depensage.engine.formatter import (
    format_for_sheet, format_bank_expenses_for_sheet, format_income_for_sheet,
)
from depensage.engine.carryover import run_carryover, get_previous_month
from depensage.engine.staging import StagedPipelineResult, RowMeta
from depensage.sheets.spreadsheet_handler import SheetHandler, SECTION_MARKERS
from depensage.sheets.sheet_utils import SheetUtils

logger = logging.getLogger(__name__)


@dataclass
class MonthResult:
    month: str
    year: int
    written: int
    duplicates: int
    income_written: int = 0
    income_duplicates: int = 0


@dataclass
class PipelineResult:
    total_parsed: int
    in_process_skipped: int
    classified: int
    unclassified: int
    months: list[MonthResult] = field(default_factory=list)
    unclassified_merchants: list[str] = field(default_factory=list)
    cc_lump_sums: list[CCLumpSum] = field(default_factory=list)


def run_pipeline(statement_paths, handlers, classifier, year=None,
                 bank_classifier=None, income_classifier=None):
    """Run the full expense and income processing pipeline.

    Returns a StagedPipelineResult with all writes staged (not yet committed).
    The caller should call .commit(handlers) to execute the writes.

    Args:
        statement_paths: List of file paths (CC statements and/or bank transcripts).
        handlers: Dict mapping year (int) to authenticated SheetHandler,
                  or a single SheetHandler (used for all years).
        classifier: LookupClassifier instance (for CC transactions).
        year: Optional year filter (int). If set, only transactions
              from this year are processed.
        bank_classifier: BankLookupClassifier instance (for bank expenses).
        income_classifier: IncomeLookupClassifier instance (for bank income).

    Returns:
        StagedPipelineResult with processing statistics and staged writes.

    Raises:
        ValueError: If marker not found or no handler for a year.
    """
    parser = StatementParser()

    # Normalize handlers
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

    # 1. Parse all files, auto-detecting format
    cc_dfs = []
    bank_expenses = []
    bank_income = []
    all_cc_lump_sums = []
    all_monthly_balances = {}  # (year, month) -> balance

    for path in statement_paths:
        if detect_bank_transcript(path):
            bank_result = parse_bank_transcript(path)
            if bank_result:
                if not bank_result.expenses.empty:
                    bank_expenses.append(bank_result.expenses)
                if not bank_result.income.empty:
                    bank_income.append(bank_result.income)
                all_cc_lump_sums.extend(bank_result.cc_lump_sums)
                # Merge monthly balances (later file wins on conflict)
                all_monthly_balances.update(bank_result.monthly_balances)
        else:
            df = parser.parse_statement(path)
            if df is not None and not df.empty:
                cc_dfs.append(df)

    # 2. Process CC transactions
    cc_result = _process_cc(cc_dfs, parser, classifier, year)

    # 3. Process bank expenses
    bank_exp_result = _process_bank_expenses(
        bank_expenses, bank_classifier, year
    )

    # 4. Process bank income
    bank_inc_result = _process_bank_income(
        bank_income, income_classifier, year
    )

    total_parsed = cc_result.total_parsed + bank_exp_result.total + bank_inc_result.total

    # Build staged result
    staged = StagedPipelineResult(
        total_parsed=total_parsed,
        in_process_skipped=cc_result.in_process_skipped,
        classified=(
            cc_result.classified + bank_exp_result.classified
            + bank_inc_result.classified
        ),
        unclassified=(
            cc_result.unclassified + bank_exp_result.unclassified
            + bank_inc_result.unclassified
        ),
        unclassified_merchants=cc_result.unclassified_merchants,
        cc_lump_sums=all_cc_lump_sums,
    )

    # 5. Merge all expenses and stage by month
    all_expenses = []
    if cc_result.cc_expenses is not None and not cc_result.cc_expenses.empty:
        all_expenses.append(("cc", cc_result.cc_expenses))
    if bank_exp_result.expenses is not None and not bank_exp_result.expenses.empty:
        all_expenses.append(("bank", bank_exp_result.expenses))

    sheets_seen = set()

    def _try_carryover(sheet_name, tx_year):
        prev_month, year_offset = get_previous_month(sheet_name)
        if prev_month is None:
            return
        prev_year = tx_year + year_offset
        try:
            prev_handler = get_handler(prev_year)
        except ValueError:
            logger.info(
                f"No handler for year {prev_year}, skipping carryover "
                f"from {prev_month}"
            )
            return
        if not prev_handler.sheet_exists(prev_month):
            logger.info(
                f"Previous month sheet '{prev_month}' does not exist, "
                f"skipping carryover"
            )
            return
        dest_handler = get_handler(tx_year)
        result = run_carryover(
            prev_handler, prev_month, dest_handler, sheet_name
        )
        logger.info(
            f"Carryover {prev_month} → {sheet_name}: "
            f"{result['budget_lines']} budget, "
            f"{result['savings_lines']} savings"
        )

    def _ensure_sheet(sample_date, tx_year):
        """Get or create month sheet, run carryover if new."""
        handler = get_handler(tx_year)
        expected_name = SheetUtils.get_sheet_name_for_date(sample_date)
        is_new = expected_name and not handler.sheet_exists(expected_name)
        sheet_name = handler.get_or_create_month_sheet(sample_date)
        if not sheet_name:
            return None
        if is_new and sheet_name not in sheets_seen:
            sheets_seen.add(sheet_name)
            _try_carryover(sheet_name, tx_year)
        return sheet_name

    # Stage expenses (CC + bank)
    for source, expenses_df in all_expenses:
        expenses_df = expenses_df.copy()
        expenses_df["_year_month"] = expenses_df["date"].dt.to_period("M")

        for period, group in expenses_df.groupby("_year_month"):
            sample_date = group["date"].iloc[0]
            tx_year = sample_date.year
            handler = get_handler(tx_year)

            sheet_name = _ensure_sheet(sample_date, tx_year)
            if not sheet_name:
                logger.error(f"Failed to get/create sheet for {period}")
                continue

            marker_row = handler.find_section_marker(sheet_name, "budget")
            if marker_row is None:
                raise ValueError(
                    f"No {SECTION_MARKERS['budget']} marker found in "
                    f"sheet '{sheet_name}'."
                )

            existing_rows = handler.read_expense_rows(sheet_name)
            month_txns = group.drop(columns=["_year_month"])
            new_txns = deduplicate(month_txns, existing_rows)
            dup_count = len(month_txns) - len(new_txns)

            stage = staged.get_or_create_stage(sheet_name, tx_year)
            stage.duplicates += dup_count

            if new_txns.empty:
                continue

            if source == "cc":
                formatted = format_for_sheet(new_txns)
            else:
                formatted = format_bank_expenses_for_sheet(new_txns)

            first_empty = handler.find_first_empty_expense_row(sheet_name)
            rows_needed = len(formatted)
            last_data_row = marker_row - 2
            available = last_data_row - first_empty + 1

            insert_needed = max(0, rows_needed - available)

            # Build RowMeta for each formatted row
            for row in formatted:
                cat = row[4]   # category
                sub = row[2]   # subcategory
                stage.expense_meta.append(RowMeta(
                    orig_category=cat, orig_subcategory=sub,
                    needs_review=not cat,
                ))

            stage.new_expenses.extend(formatted)
            stage.expense_start_row = first_empty
            stage.expense_insert_needed += insert_needed

    # Stage income
    if bank_inc_result.income is not None and not bank_inc_result.income.empty:
        income_df = bank_inc_result.income.copy()
        income_df["_year_month"] = income_df["date"].dt.to_period("M")

        for period, group in income_df.groupby("_year_month"):
            sample_date = group["date"].iloc[0]
            tx_year = sample_date.year
            handler = get_handler(tx_year)

            sheet_name = _ensure_sheet(sample_date, tx_year)
            if not sheet_name:
                logger.error(f"Failed to get/create sheet for income {period}")
                continue

            # Dedup income
            existing_income = handler.read_income_rows(sheet_name)
            month_income = group.drop(columns=["_year_month"])
            new_income = deduplicate_income(month_income, existing_income)
            inc_dup_count = len(month_income) - len(new_income)

            stage = staged.get_or_create_stage(sheet_name, tx_year)
            stage.income_duplicates += inc_dup_count

            if new_income.empty:
                continue

            formatted = format_income_for_sheet(new_income)

            # Find insertion point in income section
            first_empty = handler.find_first_empty_income_row(sheet_name)
            if first_empty is None:
                logger.error(
                    f"Could not find income insertion point in {sheet_name}"
                )
                continue

            savings_marker = handler.find_section_marker(sheet_name, "savings")
            if savings_marker is None:
                logger.error(f"No savings marker in {sheet_name}")
                continue

            last_income_data = savings_marker - 3
            available = last_income_data - first_empty + 1
            insert_needed = max(0, len(formatted) - available)

            # Build RowMeta for income rows
            for row in formatted:
                cat = row[2]   # category
                sub = row[0]   # comments (used as subcategory for income)
                stage.income_meta.append(RowMeta(
                    orig_category=cat, orig_subcategory=sub,
                    needs_review=not cat,
                ))

            stage.new_income.extend(formatted)
            stage.income_start_row = first_empty
            stage.income_insert_needed += insert_needed

    # 6. Stage bank balances
    for (bal_year, bal_month), balance in all_monthly_balances.items():
        if year is not None and bal_year != year:
            continue
        month_names = {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December",
        }
        sheet_name = month_names.get(bal_month)
        if not sheet_name:
            continue
        try:
            handler = get_handler(bal_year)
        except ValueError:
            logger.info(f"No handler for year {bal_year}, skipping bank balance")
            continue
        if not handler.sheet_exists(sheet_name):
            logger.info(
                f"Sheet '{sheet_name}' does not exist, skipping bank balance"
            )
            continue
        stage = staged.get_or_create_stage(sheet_name, bal_year)
        stage.bank_balance = balance
        logger.info(
            f"Staged bank balance {balance:,.2f} for {sheet_name} {bal_year}"
        )

    return staged


def _month_order(month_name):
    """Return sort key for English month names."""
    months = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
    }
    return months.get(month_name, 0)


@dataclass
class _CCResult:
    total_parsed: int
    in_process_skipped: int
    classified: int
    unclassified: int
    cc_expenses: pd.DataFrame
    unclassified_merchants: list


def _process_cc(cc_dfs, parser, classifier, year):
    """Process CC statement DataFrames through classification."""
    if not cc_dfs:
        return _CCResult(0, 0, 0, 0, pd.DataFrame(), [])

    all_cc = pd.concat(cc_dfs, ignore_index=True).sort_values("date")
    total = len(all_cc)

    filtered = StatementParser.filter_in_process(all_cc)
    skipped = total - len(filtered)

    if filtered.empty:
        return _CCResult(total, skipped, 0, 0, pd.DataFrame(), [])

    if year is not None:
        filtered = filtered[filtered["date"].dt.year == year].reset_index(drop=True)
        if filtered.empty:
            return _CCResult(total, skipped, 0, 0, pd.DataFrame(), [])

    result = classifier.classify(filtered)
    unclassified_merchants = (
        result.unclassified["business_name"].unique().tolist()
        if not result.unclassified.empty else []
    )

    if not result.unclassified.empty:
        unc = result.unclassified.copy()
        unc["category"] = ""
        unc["subcategory"] = ""
        combined = pd.concat(
            [result.classified, unc], ignore_index=True
        ).sort_values("date")
    else:
        combined = result.classified

    return _CCResult(
        total_parsed=total,
        in_process_skipped=skipped,
        classified=len(result.classified),
        unclassified=len(result.unclassified),
        cc_expenses=combined,
        unclassified_merchants=unclassified_merchants,
    )


@dataclass
class _BankExpResult:
    total: int
    classified: int
    unclassified: int
    expenses: pd.DataFrame


def _process_bank_expenses(bank_dfs, bank_classifier, year):
    """Process bank expense DataFrames through classification."""
    if not bank_dfs:
        return _BankExpResult(0, 0, 0, pd.DataFrame())

    all_bank = pd.concat(bank_dfs, ignore_index=True).sort_values("date")

    if year is not None:
        all_bank = all_bank[all_bank["date"].dt.year == year].reset_index(drop=True)

    total = len(all_bank)

    if total == 0 or bank_classifier is None:
        # Without classifier, write with empty categories
        if total > 0:
            all_bank["category"] = ""
            all_bank["subcategory"] = ""
        return _BankExpResult(total, 0, total, all_bank)

    result = bank_classifier.classify(all_bank)

    if not result.unclassified.empty:
        unc = result.unclassified.copy()
        unc["category"] = ""
        unc["subcategory"] = ""
        combined = pd.concat(
            [result.classified, unc], ignore_index=True
        ).sort_values("date")
    else:
        combined = result.classified

    return _BankExpResult(
        total=total,
        classified=len(result.classified),
        unclassified=len(result.unclassified),
        expenses=combined,
    )


@dataclass
class _BankIncResult:
    total: int
    classified: int
    unclassified: int
    income: pd.DataFrame


def _process_bank_income(income_dfs, income_classifier, year):
    """Process bank income DataFrames through classification."""
    if not income_dfs:
        return _BankIncResult(0, 0, 0, pd.DataFrame())

    all_income = pd.concat(income_dfs, ignore_index=True).sort_values("date")

    if year is not None:
        all_income = all_income[
            all_income["date"].dt.year == year
        ].reset_index(drop=True)

    total = len(all_income)

    if total == 0 or income_classifier is None:
        if total > 0:
            all_income["category"] = ""
            all_income["comments"] = all_income["action"]
        return _BankIncResult(total, 0, total, all_income)

    result = income_classifier.classify(all_income)

    if not result.unclassified.empty:
        unc = result.unclassified.copy()
        unc["category"] = ""
        unc["comments"] = unc["action"]
        combined = pd.concat(
            [result.classified, unc], ignore_index=True
        ).sort_values("date")
    else:
        combined = result.classified

    return _BankIncResult(
        total=total,
        classified=len(result.classified),
        unclassified=len(result.unclassified),
        income=combined,
    )
