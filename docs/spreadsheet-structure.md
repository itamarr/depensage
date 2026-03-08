# Spreadsheet Structure

The spreadsheet is entirely in Hebrew.

## Monthly Sheets (January–December)

Each month has ~130 expense rows with columns (right-to-left):
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

## Green Color-Coding (Charged vs. Pending)

Expenses highlighted green have been charged to the bank account. Non-green expenses are still pending on the credit card. This distinction is structurally important — the reconciliation section only works correctly when charged expenses are marked.

A custom **Apps Script `sumbycolor` function** sums green-highlighted cells. This function must be manually triggered by editing its cell.

**Status**: This color-based system is fragile and slated for replacement with a proper status mechanism (see roadmap Phase 4).

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
