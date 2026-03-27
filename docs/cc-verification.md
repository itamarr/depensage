# CC Verification Logic

## Goal

Compare CC lump sums charged to the bank account against the sum of CC-tagged
expenses in the spreadsheet, to verify nothing was missed or double-counted.

## How CC Billing Works

- Each month the bank debits 2 CC lump sums (one per card — user + wife)
- Each card has its own billing cutoff (Itamar's: day 8, Noa's: varies up to day 10)
- The `charge_date` column in the CC statement is the ground truth: it's the exact
  date the CC company debits the bank account
- Status is set by comparing `charge_date` month to transaction month:
  - Same month → **CC** (charged this cycle)
  - Different month → empty (pending, charged next cycle)

## Data Sources

- **CC lump sums**: Extracted from bank transcript by `bank_parser.py`
  (action contains "כרטיסי אשראי ל"). Stored in `PipelineResult.cc_lump_sums`.
- **CC expenses**: Rows in the spreadsheet with status "CC" (charged) or empty
  status (pending). Status determined from `charge_date` during formatting.

## Verification Algorithm

For each month with CC lump sums:

1. Read current month's **charged** expenses (H = "CC")
2. Read previous month's **pending** expenses (H = "" and source is CC, not BANK)
3. Sum = charged_total + pending_total
4. Compare against the CC lump sum(s) for this month
5. Since there are 2 cards (user + wife), and we may only have one card's
   data ingested, match against whichever single lump sum fits. If none fits,
   try the sum of both cards. Flag mismatches for debugging.

## Notes

- Wife's CC: once ingested, both lump sums should be accounted for.
- Tolerance: small rounding differences (< 0.05 NIS) are accepted.
