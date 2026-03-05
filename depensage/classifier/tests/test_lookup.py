"""Tests for the lookup-based classifier."""

import json
import os
import tempfile
import unittest

import pandas as pd

from depensage.classifier.lookup import LookupClassifier, Classification


class TestLookupClassifier(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.lookup_path = os.path.join(self.tmpdir, "lookup.json")
        self.sample_lookup = {
            "exact": {
                "רמי לוי 123": {"category": "סופר", "subcategory": ""},
                "הוט מובייל": {"category": "חשבונות", "subcategory": "סלולר"},
            },
            "patterns": [
                {"prefix": "ALIEXPRESS", "category": "שונות", "subcategory": "אונליין"},
            ],
        }
        with open(self.lookup_path, "w", encoding="utf-8") as f:
            json.dump(self.sample_lookup, f, ensure_ascii=False)

        self.classifier = LookupClassifier(lookup_path=self.lookup_path)

    def tearDown(self):
        if os.path.exists(self.lookup_path):
            os.remove(self.lookup_path)
        os.rmdir(self.tmpdir)

    def test_load(self):
        self.assertEqual(len(self.classifier.exact), 2)
        self.assertEqual(len(self.classifier.patterns), 1)

    def test_load_missing_file(self):
        classifier = LookupClassifier(lookup_path="/nonexistent/path.json")
        self.assertEqual(len(classifier.exact), 0)
        self.assertEqual(len(classifier.patterns), 0)

    def test_exact_match(self):
        result = self.classifier.match("רמי לוי 123")
        self.assertEqual(result.category, "סופר")
        self.assertEqual(result.subcategory, "")

    def test_pattern_match(self):
        result = self.classifier.match("ALIEXPRESS7382912")
        self.assertEqual(result.category, "שונות")
        self.assertEqual(result.subcategory, "אונליין")

    def test_no_match(self):
        result = self.classifier.match("חנות חדשה לגמרי")
        self.assertIsNone(result)

    def test_exact_takes_priority_over_pattern(self):
        self.classifier.add_exact("ALIEXPRESS7382912", "בילויים וביזבוזים", "גאדג'טים")
        result = self.classifier.match("ALIEXPRESS7382912")
        self.assertEqual(result.category, "בילויים וביזבוזים")

    def test_classify_dataframe(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-02-01", "2024-02-02", "2024-02-03"]),
            "business_name": ["רמי לוי 123", "ALIEXPRESS999", "חנות חדשה"],
            "amount": [150.0, 30.0, 200.0],
        })

        result = self.classifier.classify(df)

        self.assertEqual(len(result.classified), 2)
        self.assertEqual(len(result.unclassified), 1)
        self.assertEqual(result.classified.iloc[0]["category"], "סופר")
        self.assertEqual(result.classified.iloc[1]["category"], "שונות")
        self.assertEqual(result.unclassified.iloc[0]["business_name"], "חנות חדשה")

    def test_add_exact_and_save(self):
        self.classifier.add_exact("חנות חדשה", "שונות", "כללי")

        # Verify in memory
        result = self.classifier.match("חנות חדשה")
        self.assertEqual(result.category, "שונות")

        # Verify persisted
        reloaded = LookupClassifier(lookup_path=self.lookup_path)
        result = reloaded.match("חנות חדשה")
        self.assertEqual(result.category, "שונות")

    def test_add_pattern_and_save(self):
        self.classifier.add_pattern("AMAZON", "שונות", "אונליין")

        result = self.classifier.match("AMAZON IL 12345")
        self.assertEqual(result.category, "שונות")

        reloaded = LookupClassifier(lookup_path=self.lookup_path)
        result = reloaded.match("AMAZON IL 12345")
        self.assertEqual(result.category, "שונות")

    def test_save_roundtrip(self):
        self.classifier.save()
        with open(self.lookup_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("רמי לוי 123", data["exact"])
        self.assertEqual(len(data["patterns"]), 1)
        self.assertEqual(data["patterns"][0]["prefix"], "ALIEXPRESS")

    def test_pattern_match_case_insensitive(self):
        result = self.classifier.match("aliexpress7382912")
        self.assertEqual(result.category, "שונות")

        result = self.classifier.match("Aliexpress7382912")
        self.assertEqual(result.category, "שונות")

    def test_match_empty_or_nan(self):
        self.assertIsNone(self.classifier.match(""))
        self.assertIsNone(self.classifier.match(None))
        self.assertIsNone(self.classifier.match(float("nan")))

    def test_classify_with_empty_business_names(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-02-01", "2024-02-02", "2024-02-03"]),
            "business_name": ["רמי לוי 123", "", "הוט מובייל"],
            "amount": [150.0, 30.0, 200.0],
        })

        result = self.classifier.classify(df)
        self.assertEqual(len(result.classified), 2)
        self.assertEqual(len(result.unclassified), 1)

    def test_classify_empty_dataframe(self):
        df = pd.DataFrame(columns=["date", "business_name", "amount"])
        result = self.classifier.classify(df)
        self.assertEqual(len(result.classified), 0)
        self.assertEqual(len(result.unclassified), 0)


if __name__ == "__main__":
    unittest.main()
