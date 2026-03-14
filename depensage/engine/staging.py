"""
Staging layer for the expense pipeline.

Collects all pending writes into MonthStage objects. The caller
can inspect, display, export to XLSX, and then commit (or discard).
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime

from depensage.engine.bank_parser import CCLumpSum

logger = logging.getLogger(__name__)


@dataclass
class MonthStage:
    """Staged writes for a single month sheet."""
    month: str
    year: int
    new_expenses: list[list] = field(default_factory=list)  # B:H rows
    new_income: list[list] = field(default_factory=list)    # D:G rows
    expense_start_row: int = 0
    income_start_row: int | None = None
    expense_insert_needed: int = 0
    income_insert_needed: int = 0
    duplicates: int = 0
    income_duplicates: int = 0


def _month_order(month_name):
    """Return sort key for English month names."""
    months = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
    }
    return months.get(month_name, 0)


@dataclass
class StagedPipelineResult:
    """Pipeline output with all writes staged (not yet committed)."""
    # Statistics
    total_parsed: int = 0
    in_process_skipped: int = 0
    classified: int = 0
    unclassified: int = 0
    unclassified_merchants: list[str] = field(default_factory=list)
    cc_lump_sums: list[CCLumpSum] = field(default_factory=list)

    # Staged writes
    month_stages: dict[tuple, MonthStage] = field(default_factory=dict)

    def get_or_create_stage(self, sheet_name, year):
        """Get or create a MonthStage for the given month/year."""
        key = (sheet_name, year)
        if key not in self.month_stages:
            self.month_stages[key] = MonthStage(month=sheet_name, year=year)
        return self.month_stages[key]

    def sorted_stages(self):
        """Return month stages sorted by year then month."""
        return sorted(
            self.month_stages.values(),
            key=lambda s: (s.year, _month_order(s.month)),
        )

    def summary(self):
        """Human-readable summary for CLI stdout."""
        lines = [
            f"  Parsed:     {self.total_parsed}",
            f"  In-process: {self.in_process_skipped} (skipped)",
            f"  Classified: {self.classified}",
            f"  Unknown:    {self.unclassified}",
            "",
        ]

        for stage in self.sorted_stages():
            parts = []
            if stage.new_expenses:
                parts.append(f"{len(stage.new_expenses)} expenses")
            if stage.duplicates:
                parts.append(f"{stage.duplicates} duplicates")
            if stage.new_income:
                parts.append(f"{len(stage.new_income)} income")
            if stage.income_duplicates:
                parts.append(f"{stage.income_duplicates} income dupes")
            if parts:
                lines.append(f"  {stage.month} {stage.year}: {', '.join(parts)}")

        if self.unclassified_merchants:
            lines.append(f"\n  Unclassified merchants ({len(self.unclassified_merchants)}):")
            for name in self.unclassified_merchants:
                lines.append(f"    - {name}")

        if self.cc_lump_sums:
            total = sum(ls.amount for ls in self.cc_lump_sums)
            lines.append(
                f"\n  CC lump sums from bank ({len(self.cc_lump_sums)}): "
                f"{total:,.2f}"
            )
            for ls in self.cc_lump_sums:
                date_str = (
                    ls.date.strftime("%Y-%m-%d")
                    if hasattr(ls.date, "strftime") else str(ls.date)
                )
                lines.append(f"    - {ls.amount:,.2f} ({date_str})")

        return "\n".join(lines)

    def has_writes(self):
        """Return True if there are any staged writes."""
        return any(
            stage.new_expenses or stage.new_income
            for stage in self.month_stages.values()
        )

    def export_xlsx(self, path=None):
        """Export staged changes to XLSX for review.

        Args:
            path: Output path. Defaults to .artifacts/staged_<timestamp>.xlsx.

        Returns:
            File path of the exported XLSX.
        """
        import openpyxl
        from openpyxl.styles import Font, Alignment

        if path is None:
            os.makedirs(".artifacts", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f".artifacts/staged_{timestamp}.xlsx"

        wb = openpyxl.Workbook()
        # Create summary sheet first
        summary_ws = wb.active
        summary_ws.title = "Summary"
        summary_ws.sheet_view.rightToLeft = True

        bold = Font(bold=True)
        summary_ws.append(["חודש", "הוצאות חדשות", "כפילויות", "הכנסות חדשות", "כפילויות הכנסות"])
        for cell in summary_ws[1]:
            cell.font = bold

        for stage in self.sorted_stages():
            summary_ws.append([
                f"{stage.month} {stage.year}",
                len(stage.new_expenses),
                stage.duplicates,
                len(stage.new_income),
                stage.income_duplicates,
            ])

        # Add unclassified merchants to summary
        if self.unclassified_merchants:
            summary_ws.append([])
            summary_ws.append(["סוחרים לא מסווגים"])
            summary_ws[summary_ws.max_row][0].font = bold
            for name in self.unclassified_merchants:
                summary_ws.append([name])

        # Per-month sheets with new rows
        expense_headers = ["שם בית עסק", "הערות", "תת קטגוריה", "כמה", "קטגוריה", "תאריך", "סטטוס"]
        income_headers = ["הערות", "כמה", "קטגוריה", "תאריך"]

        for stage in self.sorted_stages():
            if not stage.new_expenses and not stage.new_income:
                continue

            ws_name = f"{stage.month[:3]} {stage.year}"
            ws = wb.create_sheet(title=ws_name)
            ws.sheet_view.rightToLeft = True

            if stage.new_expenses:
                ws.append(expense_headers)
                for cell in ws[1]:
                    cell.font = bold
                for row in stage.new_expenses:
                    ws.append(row)

            if stage.new_income:
                if stage.new_expenses:
                    ws.append([])  # blank separator
                ws.append(income_headers)
                header_row = ws.max_row
                for cell in ws[header_row]:
                    cell.font = bold
                for row in stage.new_income:
                    ws.append(row)

        wb.save(path)
        return path

    def commit(self, handlers):
        """Execute all staged writes to the spreadsheet.

        Args:
            handlers: Dict mapping year (int) to SheetHandler,
                      or a single SheetHandler.

        Returns:
            Committed PipelineResult with final statistics.
        """
        from depensage.engine.pipeline import MonthResult, PipelineResult
        from depensage.sheets.spreadsheet_handler import SECTION_MARKERS

        if not isinstance(handlers, dict):
            single = handlers
            get_handler = lambda y: single
        else:
            def get_handler(y):
                return handlers[y]

        month_results = []
        for stage in self.sorted_stages():
            handler = get_handler(stage.year)
            written = 0
            income_written = 0

            # Write expenses
            if stage.new_expenses:
                if stage.expense_insert_needed > 0:
                    marker = handler.find_section_marker(stage.month, "budget")
                    if marker:
                        handler.insert_rows(
                            stage.month, marker - 1, stage.expense_insert_needed
                        )
                handler.write_expense_rows(
                    stage.month, stage.expense_start_row, stage.new_expenses
                )
                written = len(stage.new_expenses)

            # Write income
            if stage.new_income and stage.income_start_row is not None:
                if stage.income_insert_needed > 0:
                    savings_marker = handler.find_section_marker(
                        stage.month, "savings"
                    )
                    if savings_marker:
                        handler.insert_rows(
                            stage.month, savings_marker - 2,
                            stage.income_insert_needed,
                        )
                handler.write_income_rows(
                    stage.month, stage.income_start_row, stage.new_income
                )
                income_written = len(stage.new_income)

            month_results.append(MonthResult(
                month=stage.month,
                year=stage.year,
                written=written,
                duplicates=stage.duplicates,
                income_written=income_written,
                income_duplicates=stage.income_duplicates,
            ))

        return PipelineResult(
            total_parsed=self.total_parsed,
            in_process_skipped=self.in_process_skipped,
            classified=self.classified,
            unclassified=self.unclassified,
            months=month_results,
            unclassified_merchants=self.unclassified_merchants,
            cc_lump_sums=self.cc_lump_sums,
        )
