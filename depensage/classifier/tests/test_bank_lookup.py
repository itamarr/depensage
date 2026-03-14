"""
Unit tests for bank and income lookup classifiers.
"""

import json
import os
import tempfile
import unittest

import pandas as pd

from depensage.classifier.bank_lookup import BankLookupClassifier, parse_details
from depensage.classifier.income_lookup import IncomeLookupClassifier


class TestParseDetails(unittest.TestCase):

    def test_outgoing_transfer(self):
        parsed = parse_details("לטובת: ועד הבית הרצל 1 עבור: תשלום חודשי")
        self.assertEqual(parsed["recipient"], "ועד הבית הרצל 1")
        self.assertEqual(parsed["purpose"], "תשלום חודשי")
        self.assertEqual(parsed["sender"], "")

    def test_incoming_transfer(self):
        parsed = parse_details("המבצע: ישראל ישראלי עבור: מתנה")
        self.assertEqual(parsed["sender"], "ישראל ישראלי")
        self.assertEqual(parsed["purpose"], "מתנה")
        self.assertEqual(parsed["recipient"], "")

    def test_purpose_only(self):
        parsed = parse_details("עבור: נעה ישראלי מזהה 123456")
        self.assertEqual(parsed["purpose"], "נעה ישראלי")
        self.assertEqual(parsed["recipient"], "")
        self.assertEqual(parsed["sender"], "")

    def test_empty_details(self):
        parsed = parse_details("")
        self.assertEqual(parsed, {"recipient": "", "sender": "", "purpose": ""})

    def test_none_details(self):
        parsed = parse_details(None)
        self.assertEqual(parsed, {"recipient": "", "sender": "", "purpose": ""})


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

    def test_details_match_recipient_only(self):
        """Details matching on recipient field only."""
        data = {
            "exact": [],
            "patterns": [],
            "details_matches": [
                {"recipient": "ועד הבית", "category": "חשבונות",
                 "subcategory": "ועד בית", "business_name": "ועד בית"},
            ],
        }
        with open(self.lookup_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        classifier = BankLookupClassifier(self.lookup_path)

        # Match via recipient in details
        result = classifier.classify_one(
            "העב' לאחר-נייד",
            details="לטובת: ועד הבית הרצל 1 עבור: תשלום חודשי",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "חשבונות")
        self.assertEqual(result.subcategory, "ועד בית")

        # No details → no match
        result = classifier.classify_one("העב' לאחר-נייד")
        self.assertIsNone(result)

    def test_details_match_recipient_and_purpose(self):
        """Details matching requires both recipient and purpose when specified."""
        data = {
            "exact": [],
            "patterns": [],
            "details_matches": [
                {"recipient": "ישראל ישראלי", "purpose": "ריהוט",
                 "category": "בית", "subcategory": "",
                 "business_name": "ישראל ישראלי"},
            ],
        }
        with open(self.lookup_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        classifier = BankLookupClassifier(self.lookup_path)

        # Both fields match → classified
        result = classifier.classify_one(
            "העב' לאחר-נייד",
            details="לטובת: ישראל ישראלי עבור: ריהוט",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "בית")

        # Recipient matches but purpose doesn't → not classified
        result = classifier.classify_one(
            "העב' לאחר-נייד",
            details="לטובת: ישראל ישראלי עבור: חשמל",
        )
        self.assertIsNone(result)

    def test_details_match_in_dataframe(self):
        """Details matching works through the classify() DataFrame method."""
        data = {
            "exact": [],
            "patterns": [],
            "details_matches": [
                {"recipient": "ועד הבית", "category": "חשבונות",
                 "subcategory": "ועד בית", "business_name": "ועד בית"},
            ],
        }
        with open(self.lookup_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        classifier = BankLookupClassifier(self.lookup_path)

        df = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-22"]),
            "action": ["העב' לאחר-נייד"],
            "details": ["לטובת: ועד הבית הרצל 1 עבור: תשלום"],
            "amount": [600],
            "reference": [""],
        })
        result = classifier.classify(df)
        self.assertEqual(len(result.classified), 1)
        self.assertEqual(result.classified.iloc[0]["category"], "חשבונות")

    def test_action_match_takes_priority_over_details(self):
        """Action-based matching should take priority over details matching."""
        data = {
            "exact": [
                {"action": "מכבי", "category": "בריאות",
                 "subcategory": "מכבי", "business_name": "מכבי"},
            ],
            "patterns": [],
            "details_matches": [
                {"recipient": "מכבי", "category": "שונות",
                 "subcategory": "", "business_name": ""},
            ],
        }
        with open(self.lookup_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        classifier = BankLookupClassifier(self.lookup_path)

        result = classifier.classify_one(
            "מכבי", details="לטובת: מכבי עבור: something")
        self.assertEqual(result.category, "בריאות")

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

    def test_details_match_sender(self):
        """Details matching on sender field for income."""
        data = {
            "exact": [],
            "patterns": [],
            "details_matches": [
                {"sender": "ישראל ישראלי", "category": "מתנה",
                 "comments": "מהמשפחה"},
            ],
        }
        with open(self.lookup_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        classifier = IncomeLookupClassifier(self.lookup_path)

        result = classifier.classify_one(
            "זיכוי מלאומי",
            details="המבצע: ישראל ישראלי עבור: אירוע",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "מתנה")
        self.assertEqual(result.comments, "מהמשפחה")

    def test_missing_lookup_file(self):
        classifier = IncomeLookupClassifier("/nonexistent/path.json")
        self.assertEqual(len(classifier.exact), 0)


if __name__ == "__main__":
    unittest.main()
