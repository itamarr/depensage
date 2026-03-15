"""Unit tests for the lookup updater module."""

import json
import os
import tempfile
import unittest

from depensage.classifier.cc_lookup import LookupClassifier
from depensage.classifier.bank_lookup import BankLookupClassifier
from depensage.classifier.income_lookup import IncomeLookupClassifier
from depensage.engine.lookup_updater import apply_lookup_updates
from depensage.engine.staging import RowChange


class TestApplyLookupUpdates(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cc_path = os.path.join(self.tmpdir, "cc_lookup.json")
        self.bank_path = os.path.join(self.tmpdir, "bank_lookup.json")
        self.income_path = os.path.join(self.tmpdir, "income_lookup.json")

        # Write minimal lookup files (CC uses dict for exact, others use list)
        with open(self.cc_path, "w") as f:
            json.dump({"exact": {}, "patterns": []}, f)
        for path in (self.bank_path, self.income_path):
            with open(path, "w") as f:
                json.dump({"exact": [], "patterns": []}, f)

        self.cc_cls = LookupClassifier(lookup_path=self.cc_path)
        self.bank_cls = BankLookupClassifier(self.bank_path)
        self.income_cls = IncomeLookupClassifier(self.income_path)

    def test_cc_update(self):
        changes = [RowChange(
            month="February", row_type="expense", source="cc",
            lookup_key="Shop A", old_category="", new_category="סופר",
            old_subcategory="", new_subcategory="",
        )]
        updated = apply_lookup_updates(
            changes, self.cc_cls, self.bank_cls, self.income_cls
        )
        self.assertIn("cc", updated)
        self.assertIn("Shop A", self.cc_cls.exact)
        self.assertEqual(self.cc_cls.exact["Shop A"].category, "סופר")

    def test_bank_update(self):
        changes = [RowChange(
            month="January", row_type="expense", source="bank",
            lookup_key="קופת חולים", old_category="", new_category="בריאות",
            old_subcategory="", new_subcategory="קופת חולים",
        )]
        updated = apply_lookup_updates(
            changes, self.cc_cls, self.bank_cls, self.income_cls
        )
        self.assertIn("bank", updated)
        result = self.bank_cls.classify_one("קופת חולים")
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "בריאות")
        self.assertEqual(result.subcategory, "קופת חולים")

    def test_income_update(self):
        changes = [RowChange(
            month="January", row_type="income", source="income",
            lookup_key="מעסיק", old_category="", new_category="משכורת",
            old_subcategory="", new_subcategory="מעסיק",
        )]
        updated = apply_lookup_updates(
            changes, self.cc_cls, self.bank_cls, self.income_cls
        )
        self.assertIn("income", updated)
        result = self.income_cls.classify_one("מעסיק")
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "משכורת")

    def test_empty_category_skipped(self):
        changes = [RowChange(
            month="February", row_type="expense", source="cc",
            lookup_key="Shop", old_category="", new_category="",
            old_subcategory="", new_subcategory="",
        )]
        updated = apply_lookup_updates(
            changes, self.cc_cls, self.bank_cls, self.income_cls
        )
        self.assertEqual(updated, [])

    def test_multiple_sources(self):
        changes = [
            RowChange(
                month="Jan", row_type="expense", source="cc",
                lookup_key="Shop", old_category="", new_category="סופר",
                old_subcategory="", new_subcategory="",
            ),
            RowChange(
                month="Jan", row_type="expense", source="bank",
                lookup_key="קופת חולים", old_category="", new_category="בריאות",
                old_subcategory="", new_subcategory="",
            ),
        ]
        updated = apply_lookup_updates(
            changes, self.cc_cls, self.bank_cls, self.income_cls
        )
        self.assertIn("cc", updated)
        self.assertIn("bank", updated)


if __name__ == "__main__":
    unittest.main()
