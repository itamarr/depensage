# Future Work & Pain Points

## Resolved

1. **Charged vs. pending tracking** — Resolved: use `charge_date` from CC statement (not hardcoded billing day). Status: CC (same month) or empty (next month).
2. **Bank reconciliation** — Resolved: bank transcript import handles recurring charges. E170/D170 mechanism tracks untransferred savings.
3. **Year-to-year transition** — Resolved: cross-year carryover tested end-to-end (Dec 2025 → Jan 2026).

## Remaining Pain Points

1. **Merged Expenses formula** — needs manual update each year to include new months in the ARRAYFORMULA.
2. **Savings transfer classification** — transfer-to-savings actions in bank transcript need own category so they don't inflate budget SUMIFs.
3. **Template updates** — C180 formula change (`+ E170 - D170`) needs applying to Month Template.

## Planned Features

- **Web app** — in progress (Phase 8). FastAPI + Svelte. See [docs/webapp.md](webapp.md).
- **Statistics** — average spending per category, savings fund breakdown (pie chart), month-over-month trends.
- **Automated CC download** — Israeli CC providers (Cal, Max, Isracard) may have APIs or scrapeable portals.
- **Budget alerts** — mid-month notifications when a category is approaching its budget limit.
