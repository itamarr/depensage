"""
Core automated expense pipeline.

Parses CC and bank statements, filters in-process transactions, classifies,
deduplicates against existing sheet data, and stages writes for the correct
monthly Google Sheets. Returns a StagedPipelineResult that the caller can
inspect, export, and commit.

All computation happens in-memory using VirtualMonth objects. The Google
Sheets API is used ONLY for reads during staging and writes during commit.
"""

import logging
from dataclasses import dataclass, field

import pandas as pd

from depensage.engine.statement_parser import StatementParser
from depensage.engine.bank_parser import (
    detect_bank_transcript, parse_bank_transcript, CCLumpSum,
)
from depensage.engine.dedup import deduplicate, deduplicate_income
from depensage.engine.formatter import (
    format_for_sheet, format_bank_expenses_for_sheet, format_income_for_sheet,
)
from depensage.engine.staging import StagedPipelineResult, RowMeta
from depensage.engine.virtual_month import (
    VirtualMonth, load_from_sheet, load_from_template,
)
from depensage.sheets.sheet_utils import SheetUtils

logger = logging.getLogger(__name__)

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}


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

    No writes to Google Sheets occur during staging. The handler is used
    only for reads (existing sheet data and template structure).
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

    # Phase 1: Parse all files, auto-detecting format
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
                all_monthly_balances.update(bank_result.monthly_balances)
        else:
            df = parser.parse_statement(path)
            if df is not None and not df.empty:
                cc_dfs.append(df)

    cc_result = _process_cc(cc_dfs, parser, classifier, year)
    bank_exp_result = _process_bank_expenses(
        bank_expenses, bank_classifier, year
    )
    bank_inc_result = _process_bank_income(
        bank_income, income_classifier, year
    )

    total_parsed = (
        cc_result.total_parsed + bank_exp_result.total + bank_inc_result.total
    )

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

    # Collect per-month transaction data
    from depensage.engine.carryover import (
        get_previous_month, compute_carryover, apply_carryover_to_vm,
    )
    from depensage.engine.savings_allocator import (
        SavingsGoal, allocate_savings,
    )
    from depensage.config.settings import load_settings
    try:
        settings = load_settings()
        default_goal = settings.get("default_savings_goal")
    except Exception:
        default_goal = None

    month_data = {}  # (month_name, year) -> dict

    def _ensure_month(sheet_name, tx_year):
        key = (sheet_name, tx_year)
        if key not in month_data:
            month_data[key] = {
                "cc_groups": [], "bank_groups": [],
                "income_group": None, "bank_balance": None,
            }
        return key

    # Group CC expenses by month
    if cc_result.cc_expenses is not None and not cc_result.cc_expenses.empty:
        cc_df = cc_result.cc_expenses.copy()
        cc_df["_year_month"] = cc_df["date"].dt.to_period("M")
        for _, group in cc_df.groupby("_year_month"):
            sample = group["date"].iloc[0]
            sheet_name = SheetUtils.get_sheet_name_for_date(sample)
            if sheet_name:
                key = _ensure_month(sheet_name, sample.year)
                month_data[key]["cc_groups"].append(
                    group.drop(columns=["_year_month"])
                )

    # Group bank expenses by month
    if bank_exp_result.expenses is not None and not bank_exp_result.expenses.empty:
        bank_df = bank_exp_result.expenses.copy()
        bank_df["_year_month"] = bank_df["date"].dt.to_period("M")
        for _, group in bank_df.groupby("_year_month"):
            sample = group["date"].iloc[0]
            sheet_name = SheetUtils.get_sheet_name_for_date(sample)
            if sheet_name:
                key = _ensure_month(sheet_name, sample.year)
                month_data[key]["bank_groups"].append(
                    group.drop(columns=["_year_month"])
                )

    # Group income by month
    if bank_inc_result.income is not None and not bank_inc_result.income.empty:
        inc_df = bank_inc_result.income.copy()
        inc_df["_year_month"] = inc_df["date"].dt.to_period("M")
        for _, group in inc_df.groupby("_year_month"):
            sample = group["date"].iloc[0]
            sheet_name = SheetUtils.get_sheet_name_for_date(sample)
            if sheet_name:
                key = _ensure_month(sheet_name, sample.year)
                month_data[key]["income_group"] = (
                    group.drop(columns=["_year_month"])
                )

    # Assign bank balances
    for (bal_year, bal_month), balance in all_monthly_balances.items():
        if year is not None and bal_year != year:
            continue
        sheet_name = MONTH_NAMES.get(bal_month)
        if not sheet_name:
            continue
        try:
            get_handler(bal_year)
        except ValueError:
            continue
        key = _ensure_month(sheet_name, bal_year)
        month_data[key]["bank_balance"] = balance

    # --- Sequential per-month processing (chronological order) ---
    # Each month is fully processed (carryover, expenses, income,
    # savings) before moving to the next, so later months see
    # accurate data from earlier ones.

    vms = {}  # (month_name, year) -> VirtualMonth

    def _get_or_load_vm(month_name, tx_year):
        key = (month_name, tx_year)
        if key in vms:
            return vms[key]
        handler = get_handler(tx_year)
        if handler.sheet_exists(month_name):
            vm = load_from_sheet(handler, month_name, tx_year)
        else:
            vm = load_from_template(handler, month_name, tx_year)
        vms[key] = vm
        return vm

    sorted_keys = sorted(
        month_data.keys(), key=lambda k: (k[1], _month_order(k[0]))
    )

    for month_name, tx_year in sorted_keys:
        data = month_data[(month_name, tx_year)]
        handler = get_handler(tx_year)
        vm = _get_or_load_vm(month_name, tx_year)

        stage = staged.get_or_create_stage(month_name, tx_year)
        if vm.is_new:
            stage.is_new = True

        # --- Step 1: Carryover from previous month ---
        if _needs_carryover(vm):
            _try_carryover(
                vm, month_name, tx_year, staged, vms,
                get_handler, _get_or_load_vm,
            )

        # --- Step 2: Stage expenses ---
        if vm.budget_marker_row == 0 and (data["cc_groups"] or data["bank_groups"]):
            raise ValueError(
                f"No ---BUDGET--- marker found in sheet '{month_name}'."
            )

        for source, groups in [("cc", data["cc_groups"]),
                               ("bank", data["bank_groups"])]:
            for group in groups:
                existing_rows = vm.expense_rows
                new_txns = deduplicate(group, existing_rows)
                dup_count = len(group) - len(new_txns)
                stage.duplicates += dup_count

                if new_txns.empty:
                    continue

                if source == "cc":
                    formatted = format_for_sheet(new_txns)
                else:
                    formatted = format_bank_expenses_for_sheet(new_txns)

                first_empty = vm.first_empty_expense_row
                rows_needed = len(formatted)
                last_data_row = vm.budget_marker_row - 2
                available = last_data_row - first_empty + 1
                insert_needed = max(0, rows_needed - available)

                for row in formatted:
                    stage.expense_meta.append(RowMeta(
                        orig_category=row[4], orig_subcategory=row[2],
                        needs_review=not row[4],
                    ))

                stage.new_expenses.extend(formatted)
                stage.expense_start_row = first_empty
                stage.expense_insert_needed += insert_needed

        # --- Step 3: Stage income ---
        income_group = data["income_group"]
        if income_group is not None and not income_group.empty:
            existing_income = vm.income_rows
            new_income = deduplicate_income(income_group, existing_income)
            inc_dup_count = len(income_group) - len(new_income)
            stage.income_duplicates += inc_dup_count

            if not new_income.empty:
                formatted = format_income_for_sheet(new_income)
                first_empty = vm.first_empty_income_row
                if first_empty and first_empty > 0 and vm.savings_marker_row > 0:
                    last_income_data = vm.savings_marker_row - 3
                    available = last_income_data - first_empty + 1
                    insert_needed = max(0, len(formatted) - available)
                    stage.income_start_row = first_empty
                    stage.income_insert_needed += insert_needed

                    for row in formatted:
                        stage.income_meta.append(RowMeta(
                            orig_category=row[2], orig_subcategory=row[0],
                            needs_review=not row[2],
                        ))
                    stage.new_income.extend(formatted)

        # --- Step 4: Update VM with staged data ---
        # So the NEXT month's carryover sees accurate remaining/totals.
        _update_vm_after_staging(vm, stage)

        # --- Step 5: Bank balance ---
        if data["bank_balance"] is not None:
            stage.bank_balance = data["bank_balance"]
            logger.info(
                f"Staged bank balance {data['bank_balance']:,.2f} "
                f"for {month_name} {tx_year}"
            )

        # --- Step 6: Savings allocation ---
        budget = vm.savings_budget_value
        if budget is not None and vm.savings_lines:
            goals = [
                SavingsGoal(
                    goal_name=sl.goal_name,
                    preset_incoming=sl.incoming,
                    outgoing=sl.outgoing,
                    target=sl.target,
                    total=compute_savings_total_from_line(sl),
                    row_number=sl.row_number,
                )
                for sl in vm.savings_lines
            ]
            if goals:
                alloc_result = allocate_savings(budget, goals, default_goal)
                stage.savings_allocations = alloc_result.allocations
                stage.savings_warning = alloc_result.warning
                logger.info(
                    f"Savings allocation for {month_name}: "
                    f"budget={budget:,.2f}, "
                    f"presets={alloc_result.total_preset:,.2f}, "
                    f"surplus={alloc_result.surplus:,.2f}"
                )

    return staged


def _try_carryover(dest_vm, month_name, tx_year, staged, vms,
                   get_handler, get_or_load_vm):
    """Attempt carryover from the previous month into dest_vm."""
    from depensage.engine.carryover import (
        get_previous_month, compute_carryover, apply_carryover_to_vm,
    )

    prev_month, year_offset = get_previous_month(month_name)
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
        if (prev_month, prev_year) not in vms:
            logger.info(
                f"Previous month sheet '{prev_month}' does not exist, "
                f"skipping carryover"
            )
            return

    source_vm = get_or_load_vm(prev_month, prev_year)

    same_spreadsheet = (
        hasattr(prev_handler, 'spreadsheet_id')
        and hasattr(get_handler(tx_year), 'spreadsheet_id')
        and prev_handler.spreadsheet_id
        == get_handler(tx_year).spreadsheet_id
    )

    result = compute_carryover(source_vm, dest_vm, same_spreadsheet)
    apply_carryover_to_vm(dest_vm, result)

    # Store updates for commit
    stage = staged.get_or_create_stage(month_name, tx_year)
    if dest_vm.is_new:
        stage.is_new = True
    all_updates = list(result.budget_updates) + list(result.savings_updates)
    if result.savings_budget_update:
        all_updates.append(result.savings_budget_update)
    stage.carryover_updates = all_updates

    logger.info(
        f"Carryover {prev_month} -> {month_name}: "
        f"{len(result.budget_updates)} budget, "
        f"{len(result.savings_updates)} savings"
    )


def _update_vm_after_staging(vm, stage):
    """Update VirtualMonth with staged data so next month's carryover
    sees accurate remaining values and income totals.
    """
    # Add staged expenses to VM (strip status column for B:G format)
    for row in stage.new_expenses:
        vm.expense_rows.append(row[:6])

    # Add staged income to VM
    for row in stage.new_income:
        vm.income_rows.append(row)

    # Recompute income_total including new income
    new_income_sum = 0.0
    for row in stage.new_income:
        try:
            new_income_sum += float(row[1]) if row[1] else 0.0
        except (ValueError, TypeError):
            pass
    if vm.income_total is not None:
        vm.income_total += new_income_sum
    elif new_income_sum > 0:
        vm.income_total = new_income_sum

    # Recompute budget remaining values: subtract new expenses per category.
    # Budget lines with a subcategory match expenses with the same
    # (category, subcategory). Lines without a subcategory match ALL
    # expenses for that category (the spreadsheet SUMIF works the same way).
    if stage.new_expenses:
        # Sum expenses by exact (category, subcategory)
        expense_by_exact = {}
        # Sum expenses by category only (for budget lines without subcat)
        expense_by_cat = {}
        for row in stage.new_expenses:
            cat = str(row[4]).strip() if row[4] else ""
            sub = str(row[2]).strip() if row[2] else ""
            try:
                amt = float(row[3]) if row[3] else 0.0
            except (ValueError, TypeError):
                amt = 0.0
            expense_by_exact.setdefault((cat, sub), 0.0)
            expense_by_exact[(cat, sub)] += amt
            expense_by_cat.setdefault(cat, 0.0)
            expense_by_cat[cat] += amt

        for line in vm.budget_lines:
            if line.subcategory:
                delta = expense_by_exact.get(
                    (line.category, line.subcategory), 0.0
                )
            else:
                delta = expense_by_cat.get(line.category, 0.0)
            if delta:
                line.remaining -= delta


def _needs_carryover(vm):
    """Check if a VirtualMonth needs carryover applied.

    Returns True if accumulated columns are all empty (zero), meaning
    carryover was never written — whether the sheet is new or existing.
    """
    carry_lines = [l for l in vm.budget_lines if l.carry_flag]
    if carry_lines and all(l.accumulated == 0 for l in carry_lines):
        return True
    if vm.savings_lines and all(l.accumulated == 0 for l in vm.savings_lines):
        return True
    return False


def compute_savings_total_from_line(sl):
    """Compute savings total (C = F + E - D) from a SavingsLine."""
    return sl.accumulated + sl.incoming - sl.outgoing


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
