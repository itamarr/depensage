"""
Lookup-based classifier for bank expense transactions.

Classifies bank debits by matching the action name (הפעולה)
against a lookup table. Supports exact matches, prefix patterns,
and details-based matching for generic actions like bank transfers.
"""

import json
import logging
import os
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


def parse_details(details):
    """Parse structured fields from a bank transaction details string.

    Bank transfer details follow the pattern:
        לטובת: <recipient> עבור: <purpose>   (outgoing)
        המבצע: <sender> עבור: <purpose>      (incoming)

    Returns dict with keys: recipient, sender, purpose.
    """
    result = {"recipient": "", "sender": "", "purpose": ""}
    if not details:
        return result

    markers = [
        ("לטובת:", "recipient"),
        ("המבצע:", "sender"),
        ("עבור:", "purpose"),
    ]
    delimiters = ["לטובת:", "המבצע:", "עבור:", "מזהה"]

    for marker, key in markers:
        idx = details.find(marker)
        if idx == -1:
            continue
        start = idx + len(marker)
        value = details[start:]
        # Find the earliest next delimiter
        end = len(value)
        for delim in delimiters:
            if delim == marker:
                continue
            pos = value.find(delim)
            if pos != -1 and pos < end:
                end = pos
        result[key] = value[:end].strip()

    return result

DEFAULT_BANK_LOOKUP_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", ".artifacts", "bank_lookup.json"
)


@dataclass
class BankClassification:
    category: str
    subcategory: str


@dataclass
class BankClassificationResult:
    classified: pd.DataFrame
    unclassified: pd.DataFrame


class BankLookupClassifier:
    """Classifies bank transactions using a local lookup table.

    Supports three matching strategies in order:
    1. Exact action match
    2. Action prefix match
    3. Details keyword match — for generic actions (e.g. bank transfers)
       where the details field identifies the purpose
    """

    def __init__(self, lookup_path=None):
        self.lookup_path = lookup_path or DEFAULT_BANK_LOOKUP_PATH
        self.exact = {}
        self.patterns = []
        self.details_matches = []
        self.load()

    def load(self):
        """Load lookup table from disk."""
        self.exact = {}
        self.patterns = []
        self.details_matches = []

        if not os.path.exists(self.lookup_path):
            logger.info(
                f"No bank lookup table at {self.lookup_path}, starting empty"
            )
            return

        with open(self.lookup_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data.get("exact", []):
            self.exact[entry["action"]] = BankClassification(
                category=entry["category"],
                subcategory=entry.get("subcategory", ""),
            )

        for entry in data.get("patterns", []):
            self.patterns.append((
                entry["prefix"],
                BankClassification(
                    category=entry["category"],
                    subcategory=entry.get("subcategory", ""),
                ),
            ))

        for entry in data.get("details_matches", []):
            match_fields = {}
            for field in ("recipient", "sender", "purpose"):
                if entry.get(field):
                    match_fields[field] = entry[field]
            self.details_matches.append((
                match_fields,
                BankClassification(
                    category=entry["category"],
                    subcategory=entry.get("subcategory", ""),
                ),
            ))

        logger.info(
            f"Loaded bank lookup: {len(self.exact)} exact, "
            f"{len(self.patterns)} patterns, "
            f"{len(self.details_matches)} details matches"
        )

    def save(self):
        """Save lookup table to disk."""
        data = {
            "exact": [
                {"action": action, "category": c.category,
                 "subcategory": c.subcategory}
                for action, c in self.exact.items()
            ],
            "patterns": [
                {"prefix": prefix, "category": c.category,
                 "subcategory": c.subcategory}
                for prefix, c in self.patterns
            ],
            "details_matches": [
                {**match_fields, "category": c.category,
                 "subcategory": c.subcategory}
                for match_fields, c in self.details_matches
            ],
        }
        os.makedirs(os.path.dirname(self.lookup_path), exist_ok=True)
        with open(self.lookup_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved bank lookup to {self.lookup_path}")

    def add_exact(self, action, category, subcategory):
        """Add an exact match entry and save."""
        self.exact[action] = BankClassification(
            category=category, subcategory=subcategory,
        )
        self.save()

    def add_pattern(self, prefix, category, subcategory):
        """Add a prefix pattern entry and save."""
        self.patterns.append((
            prefix,
            BankClassification(
                category=category, subcategory=subcategory,
            ),
        ))
        self.save()

    def add_details_match(self, match_fields, category, subcategory):
        """Add a details match entry and save.

        Args:
            match_fields: dict with keys from {recipient, sender, purpose}.
                All specified fields must match for the rule to fire.
        """
        self.details_matches.append((
            match_fields,
            BankClassification(
                category=category, subcategory=subcategory,
            ),
        ))
        self.save()

    def remove_exact(self, action):
        """Remove an exact match entry and save."""
        if action in self.exact:
            del self.exact[action]
            self.save()

    def classify_one(self, action, details=""):
        """Classify a single bank transaction.

        Tries action-based matching first (exact, then prefix),
        then falls back to details keyword matching.

        Returns BankClassification or None if unknown.
        """
        action = action.strip()

        # Exact match on action
        if action in self.exact:
            return self.exact[action]

        # Prefix match on action
        for prefix, classification in self.patterns:
            if action.startswith(prefix):
                return classification

        # Details field match (for generic actions like bank transfers)
        if details:
            parsed = parse_details(details)
            for match_fields, classification in self.details_matches:
                if all(
                    match_val in parsed.get(field, "")
                    for field, match_val in match_fields.items()
                ):
                    return classification

        return None

    def classify(self, df):
        """Classify a DataFrame of bank transactions.

        Args:
            df: DataFrame with 'action' column.

        Returns:
            BankClassificationResult with classified and unclassified DataFrames.
        """
        if df is None or df.empty:
            return BankClassificationResult(
                classified=pd.DataFrame(),
                unclassified=pd.DataFrame(),
            )

        categories = []
        subcategories = []
        classified_mask = []

        for _, row in df.iterrows():
            details = str(row.get("details", "")) if "details" in df.columns else ""
            result = self.classify_one(row["action"], details=details)
            if result:
                categories.append(result.category)
                subcategories.append(result.subcategory)
                classified_mask.append(True)
            else:
                categories.append("")
                subcategories.append("")
                classified_mask.append(False)

        classified_mask = pd.Series(classified_mask, index=df.index)

        classified = df[classified_mask].copy()
        if not classified.empty:
            classified["category"] = [
                c for c, m in zip(categories, classified_mask) if m
            ]
            classified["subcategory"] = [
                s for s, m in zip(subcategories, classified_mask) if m
            ]

        unclassified = df[~classified_mask].copy()

        count = len(classified)
        total = len(df)
        logger.info(
            f"Bank classified {count}/{total} transactions, "
            f"{total - count} unknown"
        )

        return BankClassificationResult(
            classified=classified,
            unclassified=unclassified,
        )
