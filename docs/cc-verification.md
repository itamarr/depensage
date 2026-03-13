# CC Verification Logic

## Goal

Compare CC lump sums charged to the bank account against the sum of CC-tagged
expenses in the spreadsheet, to verify nothing was missed or double-counted.

## How CC Billing Works

- Billing day is the 10th of each month
- Each month the bank debits 2 CC lump sums (one per card — user + wife)
- The lump sum on month N's 10th covers:
  - Previous month's **pending** transactions (date > 10th in month N-1)
  - Current month's **charged** transactions (date <= 10th in month N)
- The cycle may be 10th-through-9th (inclusive on both ends) — TBD after
  studying real transcript data around the boundary

## Data Sources

- **CC lump sums**: Extracted from bank transcript by `bank_parser.py`
  (action contains "כרטיסי אשראי ל"). Stored in `PipelineResult.cc_lump_sums`.
- **CC expenses**: Rows in the spreadsheet with status "CC" (charged, date <= 10)
  or empty status (pending, date > 10).

## Verification Algorithm

For each month with CC lump sums:

1. Read current month's **charged** expenses (H = "CC")
2. Read previous month's **pending** expenses (H = "" and source is CC, not BANK)
3. Sum = charged_total + pending_total
4. Compare against the CC lump sum(s) for this month
5. Since there are 2 cards (user + wife), and we may only have one card's
   data ingested, match against whichever single lump sum fits. If none fits,
   flag for debugging.

## Open Questions

- Exact billing boundary: is the 10th inclusive for charged or pending?
  Will be determined by studying a fresh transcript downloaded around the 10th.
- Wife's CC: once ingested, both lump sums should be accounted for.
- Tolerance: should we allow small rounding differences?

## Implementation Plan

1. Add `verify_cc_charges(handler, month, year, cc_lump_sums)` to pipeline or
   a new `verification.py` module
2. Call it after writing expenses, using the lump sums from bank parser
3. Report match/mismatch in CLI output
4. For web app: show verification status per month
