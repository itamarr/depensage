# Spreadsheet Structure

Cell contents are in Hebrew. Sheet tab names are in **English** (January, February, ..., December). The code uses `strftime('%B')` to map dates to sheet names.

## Monthly Sheets (January–December)

Each month has ~130 expense rows with columns (right-to-left):
- **H:** סטטוס (Status, hidden) — "CC" (charged to bank), "BANK" (direct bank charge), or empty (pending CC charge). Used for reconciliation and CC verification.
- **G:** תאריך (Date) — stored as serial numbers (days since 1899-12-30)
- **F:** קטגוריה (Category) — from dropdown linked to Categories sheet
- **E:** כמה (Amount) — in NIS
- **D:** תת קטגוריה (Sub-category)
- **C:** הערות (Notes)
- **B:** שם בית עסק (Business name) — populated since March 2025

### Section Markers

Below the transactions, each monthly sheet has four sections delimited by hidden marker rows in column B:

1. **`---BUDGET---`** — SUMIFS comparing actual spend per category against budget, with accumulated carryover from prior months. Column H (hidden) has `CARRY` flags on lines that carry over surplus to the next month.
2. **`---INCOME---`** — salary and other income entries
3. **`---SAVINGS---`** — per-goal balances with targets and months-remaining
4. **`---RECONCILIATION---`** — actual bank balances vs. expected (the "gap"/פער reveals tracking errors)

The expense total row sits between the expense data and the `---BUDGET---` marker.

Migration script for sheets predating the marker system: `scripts/plant_section_markers.py`.

### Budget Section Layout (columns B–H, relative to marker)

| Column | Content |
|--------|---------|
| B | כמה נשאר (Remaining = Budget - Expense + Accumulated) |
| C | הוצאות (Expense — SUMIF from transaction rows) |
| D | תקציב (Budget amount) |
| E | צבור (Accumulated — carryover from previous month) |
| F | תת קטגוריה (Sub-category) |
| G | קטגוריה (Category) |
| H | CARRY flag (hidden column, marks lines for carryover) |

### Income Section Layout (columns D–G, relative to marker)

| Row offset | Content |
|------------|---------|
| marker + 1 | הכנסות (Income header) |
| marker + 2 | Column headers: הערות, כמה, קטגוריה, תאריך |
| marker + 3+ | Data rows |
| savings_marker - 2 | סה"כ (Total row) |

| Column | Content |
|--------|---------|
| D | הערות (Comments — source/description of income) |
| E | כמה (Amount) |
| F | קטגוריה (Category — e.g., משכורת, קצבה, מתנה, מענק, העברה, החזר) |
| G | תאריך (Date) |

### Savings Section Layout (columns A–G, relative to marker)

| Column | Content |
|--------|---------|
| A | חודשים שנשארו (Months remaining) |
| B | יעד (Target) |
| C | סה"כ (Total = Outgoing - Incoming + Accumulated) |
| D | יוצא (Outgoing) |
| E | נכנס (Incoming) |
| F | צבור (Accumulated — carryover from previous month) |
| G | קטגוריה (Goal name) |

## Charged vs. Pending Status (Column H)

Column H tracks whether an expense has been charged to the bank account:
- **"CC"** — charged via credit card (transaction date day-of-month <= billing day 10)
- **empty** — pending CC charge (date > 10, will be charged next billing cycle)
- **"BANK"** — direct bank debit (mortgage, insurance, etc.)

This replaces the legacy green color-coding system. The reconciliation section uses these statuses to match bank debits against recorded expenses.

**Legacy**: The old `sumbycolor` Apps Script function may still exist in older sheets but is no longer the primary mechanism.

## Meta Sheets

- **Categories** — master list of 14 expense categories with sub-categories (reference data for dropdowns)
- **Budget** — monthly budget per category/sub-category
- **Month Template** — skeleton duplicated to create new monthly sheets. Source of truth for CARRY flags and section markers.
- **Merged Expenses** — ARRAYFORMULA concatenating all monthly expense data (must be manually updated at year-end to include new months)
- **Expenses so far / Income so far** — annual pivot summaries with monthly averages
- **Budget Planning** — compares prior year plan vs. actuals, sets next year's budgets, models savings and compound interest projections
- **User guide** — Hebrew operational manual for maintaining the spreadsheet
- **`_B` sheets** — wife's former business (closed late 2025, not needed going forward)

## Category List (Hebrew to English)

| Hebrew | English | Notes |
|--------|---------|-------|
| חשבונות | Bills | |
| בריאות | Health | |
| אלוני | Aloni | Child |
| צ'ופי | Chuppy | Dog |
| עסק | Business | Closed late 2025 |
| שונות | Miscellaneous | Has many sub-categories with individual budgets |
| סופר | Supermarket | |
| נסיעות | Transportation | |
| בילויים וביזבוזים | Entertainment | |
| משכנתא | Mortgage | |
| שכר דירה | Rent | |
| חסכון | Savings | |
| טיפול | Therapy | |
| יוגה | Yoga | |
