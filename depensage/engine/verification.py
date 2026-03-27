"""
CC charge verification — compares CC lump sums from the bank against
CC-tagged expenses in the spreadsheet.

Each CC lump sum (debited around the 10th of the month) covers one
billing cycle: previous month's pending expenses (status empty)
plus current month's charged expenses (status "CC").

Supports per-card matching: tries individual lump sums first (single-card),
then the sum of all lump sums (both-cards).
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from depensage.engine.bank_parser import CCLumpSum
from depensage.sheets.sheet_utils import SheetUtils

logger = logging.getLogger(__name__)

_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _month_index(sheet_name):
    """Return 1-based month index for a month sheet name."""
    for i, name in enumerate(_MONTH_NAMES, 1):
        if name == sheet_name:
            return i
    return None


def _prev_month_name(sheet_name):
    """Return the previous month's sheet name."""
    idx = _month_index(sheet_name)
    if idx is None:
        return None
    prev_idx = 12 if idx == 1 else idx - 1
    return _MONTH_NAMES[prev_idx - 1]


@dataclass
class BillingCycleVerification:
    """Verification result for a single billing cycle (one month's charges)."""
    billing_month: str  # month name where lump sums were debited
    billing_year: int
    lump_sums: list[CCLumpSum]  # The CC lump sums debited this month
    lump_total: float
    charged_total: float  # Sum of CC-status expenses in billing month (date <= 10)
    pending_total: float  # Sum of pending expenses in prev month (date > 10)
    expected_total: float  # charged + pending
    difference: float  # matched amount - expected_total
    matched: bool  # Whether the totals match (within tolerance)
    match_type: str = ""  # "single-card", "both-cards", or ""
    matched_lump: CCLumpSum | None = None  # The matched lump sum (single-card)
    unmatched_lumps: list[CCLumpSum] = field(default_factory=list)


@dataclass
class VerificationResult:
    """Full CC verification result across all billing cycles."""
    cycles: list[BillingCycleVerification] = field(default_factory=list)
    all_matched: bool = True


def group_lump_sums_by_billing_month(cc_lump_sums):
    """Group CC lump sums by the month they were debited.

    Returns dict mapping (year, month_index) to list of CCLumpSum.
    """
    groups = defaultdict(list)
    for ls in cc_lump_sums:
        key = (ls.date.year, ls.date.month)
        groups[key] = groups.get(key, [])
        groups[key].append(ls)
    return groups


def _parse_expense_rows_with_status(rows):
    """Parse raw expense rows (B:H) into amounts grouped by status.

    Args:
        rows: List of rows, each [business_name, notes, subcategory,
              amount, category, date_serial, status].

    Returns:
        (charged, pending, bank) — each a list of (date, amount) tuples.
    """
    charged = []  # H = "CC"
    pending = []  # H = "" (empty, from CC)
    bank = []     # H = "BANK"

    for row in rows:
        if len(row) < 6:
            continue
        # Amount is at index 3 (column E), date at index 5 (column G)
        try:
            amount = float(row[3]) if row[3] else 0.0
        except (ValueError, TypeError):
            continue
        if amount == 0:
            continue

        date_serial = row[5]
        status = str(row[6]).strip().upper() if len(row) > 6 and row[6] else ""

        if status == "CC":
            charged.append((date_serial, amount))
        elif status == "BANK":
            bank.append((date_serial, amount))
        else:
            # Empty status = pending CC charge
            pending.append((date_serial, amount))

    return charged, pending, bank


def verify_cc_charges(handler, cc_lump_sums, year,
                      prev_year_handler=None,
                      tolerance=0.05):
    """Verify CC charges against lump sums for all billing cycles.

    For each month where CC lump sums were debited, compares:
      - Current month's "CC" (charged) expenses
      - Previous month's "" (pending) expenses
    against the lump sum amounts.

    Status is determined by the charge_date column during formatting:
    CC = charge_date in same month as transaction, empty = next month.

    Tries per-card matching first (individual lump sum), then
    both-cards matching (sum of all lump sums).

    Args:
        handler: SheetHandler for the year being verified.
        cc_lump_sums: List of CCLumpSum from bank transcript.
        year: The year being verified.
        prev_year_handler: SheetHandler for previous year (for January).
        tolerance: Acceptable difference in NIS (default 0.05).

    Returns:
        VerificationResult with per-cycle comparisons.
    """
    if not cc_lump_sums:
        return VerificationResult()

    groups = group_lump_sums_by_billing_month(cc_lump_sums)
    result = VerificationResult()

    for (lump_year, lump_month), lump_list in sorted(groups.items()):
        billing_month_name = _MONTH_NAMES[lump_month - 1]
        prev_month_name = _prev_month_name(billing_month_name)
        lump_total = sum(ls.amount for ls in lump_list)

        # Read current month's expenses with status
        charged_total = 0.0
        current_rows = handler.read_expense_rows_with_status(billing_month_name)
        if current_rows:
            charged, pending_curr, _ = _parse_expense_rows_with_status(current_rows)
            charged_total = sum(amt for _, amt in charged)

        # Read previous month's pending expenses
        pending_total = 0.0
        if prev_month_name:
            if lump_month == 1 and prev_year_handler:
                prev_rows = prev_year_handler.read_expense_rows_with_status(
                    prev_month_name
                )
            elif lump_month == 1:
                logger.info(
                    f"No previous year handler for January verification, "
                    f"skipping pending from December"
                )
                prev_rows = None
            else:
                prev_rows = handler.read_expense_rows_with_status(prev_month_name)

            if prev_rows:
                _, prev_pending, _ = _parse_expense_rows_with_status(prev_rows)
                pending_total = sum(amt for _, amt in prev_pending)

        expected_total = charged_total + pending_total

        # Try per-card matching: check each individual lump sum
        matched = False
        match_type = ""
        matched_lump = None
        unmatched_lumps = []
        difference = lump_total - expected_total

        for ls in lump_list:
            if abs(ls.amount - expected_total) <= tolerance:
                matched = True
                match_type = "single-card"
                matched_lump = ls
                difference = ls.amount - expected_total
                unmatched_lumps = [x for x in lump_list if x is not ls]
                break

        # If no single-card match, try sum of all lump sums (both-cards)
        if not matched and len(lump_list) > 1:
            if abs(lump_total - expected_total) <= tolerance:
                matched = True
                match_type = "both-cards"
                difference = lump_total - expected_total

        cycle = BillingCycleVerification(
            billing_month=billing_month_name,
            billing_year=lump_year,
            lump_sums=lump_list,
            lump_total=lump_total,
            charged_total=charged_total,
            pending_total=pending_total,
            expected_total=expected_total,
            difference=difference,
            matched=matched,
            match_type=match_type,
            matched_lump=matched_lump,
            unmatched_lumps=unmatched_lumps,
        )
        result.cycles.append(cycle)
        if not cycle.matched:
            result.all_matched = False

    return result


def format_verification_report(result):
    """Format verification result as a human-readable string."""
    if not result.cycles:
        return "  No CC lump sums to verify."

    lines = []
    for cycle in result.cycles:
        if cycle.matched:
            status = f"OK - {cycle.match_type}" if cycle.match_type else "OK"
        else:
            status = "MISMATCH"
        lines.append(
            f"  {cycle.billing_month} {cycle.billing_year}: [{status}]"
        )

        if cycle.match_type == "single-card" and cycle.matched_lump:
            ls = cycle.matched_lump
            date_str = (
                ls.date.strftime("%Y-%m-%d")
                if hasattr(ls.date, "strftime") else str(ls.date)
            )
            lines.append(
                f"    Matched lump sum: {ls.amount:,.2f} ({date_str})"
            )
        else:
            lines.append(
                f"    Bank lump sums ({len(cycle.lump_sums)}): "
                f"{cycle.lump_total:,.2f}"
            )

        lines.append(
            f"    Charged (CC):   {cycle.charged_total:,.2f}"
        )
        lines.append(
            f"    Pending (prev): {cycle.pending_total:,.2f}"
        )
        lines.append(
            f"    Expected total: {cycle.expected_total:,.2f}"
        )

        if cycle.match_type == "single-card" and cycle.unmatched_lumps:
            for ls in cycle.unmatched_lumps:
                date_str = (
                    ls.date.strftime("%Y-%m-%d")
                    if hasattr(ls.date, "strftime") else str(ls.date)
                )
                lines.append(
                    f"    Unmatched lump:  {ls.amount:,.2f} ({date_str})"
                )

        if not cycle.matched:
            lines.append(
                f"    Difference:     {cycle.difference:+,.2f}"
            )

    return "\n".join(lines)
