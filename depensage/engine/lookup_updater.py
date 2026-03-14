"""
Apply classification changes from edited staged XLSX back to lookup tables.
"""

import logging

logger = logging.getLogger(__name__)


def apply_lookup_updates(changes, cc_classifier, bank_classifier,
                         income_classifier):
    """Update lookup tables based on user edits in the staged XLSX.

    Only updates entries where the user provided a non-empty new category.
    Saves each classifier that was modified.

    Args:
        changes: list[RowChange] from import_staged_xlsx.
        cc_classifier: LookupClassifier instance.
        bank_classifier: BankLookupClassifier instance.
        income_classifier: IncomeLookupClassifier instance.
    """
    modified = {"cc": False, "bank": False, "income": False}

    for change in changes:
        if not change.new_category:
            continue

        if change.source == "cc":
            cc_classifier.add_exact(
                change.lookup_key, change.new_category, change.new_subcategory,
            )
            modified["cc"] = True
        elif change.source == "bank":
            bank_classifier.add_exact(
                change.lookup_key, change.new_category, change.new_subcategory,
            )
            modified["bank"] = True
        elif change.source == "income":
            income_classifier.add_exact(
                change.lookup_key, change.new_category,
                comments=change.new_subcategory,
            )
            modified["income"] = True

    updated = [k for k, v in modified.items() if v]
    if updated:
        logger.info(f"Updated lookup tables: {', '.join(updated)}")
    return updated
