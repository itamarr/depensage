# Future Work & Pain Points

## Remaining Pain Points

1. **Charged vs. pending tracking** — replacing the fragile green color-coding system with a reliable status mechanism. The reconciliation section depends on distinguishing charged vs. pending expenses.
2. **Bank reconciliation** — direct bank charges (הוראות קבע: mortgage, insurance, ארנונה, etc.) repeat monthly and could be auto-populated.
3. **Merged Expenses formula** — needs manual update each year to include new months in the ARRAYFORMULA.

## Planned Features

- **Web app** — the real end product. Wife-friendly interface: drop CC statement, review unknowns visually, submit. API layer over core library, authentication, deploy.
- **Automated CC download** — Israeli CC providers (Cal, Max, Isracard) may have APIs or scrapeable portals.
- **Budget alerts** — mid-month notifications when a category is approaching its budget limit.

## Design Considerations

- **Year-to-year transition**: when processing a January statement for a new year, the pipeline should automatically look up December from the previous year's spreadsheet for carryover. The code supports this via cross-year handlers and `get_previous_month` (year_offset=-1 for January). Needs real-world testing.
