"""
Unit tests for bank and income lookup classifiers.
"""

import json
import os
import tempfile
import unittest

import pandas as pd

from depensage.classifier.bank_lookup import (
    BankLookupClassifier,
    IncomeLookupClassifier,
)


class TestBankLookupClassifier(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.lookup_path = os.path.join(self.tmpdir, "bank_lookup.json")
        data = {
            "exact": [
                {"action": "פועלים-משכנתא", "category": "משכנתא",
                 "subcategory": "", "business_name": "פועלים משכנתא"},
                {"action": "מכבי", "category": "בריאות",
                 "subcategory": "מכבי", "business_name": "מכבי"},
            ],
            "patterns": [
                {"prefix": "ני\"ע", "category": "חסכון",
                 "subcategory": "", "business_name": "ניירות ערך"},
            ],
        }
        with open(self.lookup_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        self.classifier = BankLookupClassifier(self.lookup_path)

    def test_exact_match(self):
        result = self.classifier.classify_one("פועלים-משכנתא")
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "משכנתא")
        self.assertEqual(result.business_name, "פועלים משכנתא")

    def test_prefix_match(self):
        result = self.classifier.classify_one("ני\"ע-קניה-נט")
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "חסכון")

    def test_unknown_action(self):
        result = self.classifier.classify_one("unknown action")
        self.assertIsNone(result)

    def test_classify_dataframe(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-01", "2026-01-05", "2026-01-10"]),
            "action": ["פועלים-משכנתא", "מכבי", "unknown"],
            "details": ["", "", ""],
            "amount": [5000, 300, 100],
            "reference": ["", "", ""],
        })
        result = self.classifier.classify(df)
        self.assertEqual(len(result.classified), 2)
        self.assertEqual(len(result.unclassified), 1)
        self.assertEqual(result.classified.iloc[0]["category"], "משכנתא")
        self.assertEqual(result.classified.iloc[1]["category"], "בריאות")

    def test_empty_dataframe(self):
        result = self.classifier.classify(pd.DataFrame())
        self.assertEqual(len(result.classified), 0)
        self.assertEqual(len(result.unclassified), 0)

    def test_none_input(self):
        result = self.classifier.classify(None)
        self.assertEqual(len(result.classified), 0)

    def test_missing_lookup_file(self):
        classifier = BankLookupClassifier("/nonexistent/path.json")
        self.assertEqual(len(classifier.exact), 0)
        self.assertEqual(len(classifier.patterns), 0)


class TestIncomeLookupClassifier(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.lookup_path = os.path.join(self.tmpdir, "income_lookup.json")
        data = {
            "exact": [
                {"action": "מלאנוקס טכנולו", "category": "משכורת",
                 "comments": "אנבידיה"},
                {"action": "קצבת ילדים", "category": "קצבה",
                 "comments": "קצבת ילדים"},
            ],
            "patterns": [
                {"prefix": "בטוח לאומי", "category": "מענק",
                 "comments": "ביטוח לאומי"},
            ],
        }
        with open(self.lookup_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        self.classifier = IncomeLookupClassifier(self.lookup_path)

    def test_exact_match(self):
        result = self.classifier.classify_one("מלאנוקס טכנולו")
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "משכורת")
        self.assertEqual(result.comments, "אנבידיה")

    def test_prefix_match(self):
        result = self.classifier.classify_one("בטוח לאומי חד")
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "מענק")

    def test_unknown_action(self):
        result = self.classifier.classify_one("unknown")
        self.assertIsNone(result)

    def test_classify_dataframe(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-01", "2026-01-20", "2026-01-15"]),
            "action": ["מלאנוקס טכנולו", "קצבת ילדים", "unknown"],
            "details": ["emp123", "", ""],
            "amount": [25000, 115, 500],
            "reference": ["", "", ""],
        })
        result = self.classifier.classify(df)
        self.assertEqual(len(result.classified), 2)
        self.assertEqual(len(result.unclassified), 1)
        self.assertEqual(result.classified.iloc[0]["category"], "משכורת")
        self.assertEqual(result.classified.iloc[0]["comments"], "אנבידיה")

    def test_empty_dataframe(self):
        result = self.classifier.classify(pd.DataFrame())
        self.assertEqual(len(result.classified), 0)

    def test_missing_lookup_file(self):
        classifier = IncomeLookupClassifier("/nonexistent/path.json")
        self.assertEqual(len(classifier.exact), 0)


if __name__ == "__main__":
    unittest.main()
