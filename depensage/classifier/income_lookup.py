"""
Lookup-based classifier for bank income transactions.

Classifies income (credits) by matching the action name (הפעולה)
against a lookup table. Supports exact matches and prefix patterns.
"""

import json
import logging
import os
from dataclasses import dataclass

import pandas as pd

from depensage.classifier.bank_lookup import BankClassificationResult, parse_details

logger = logging.getLogger(__name__)

DEFAULT_INCOME_LOOKUP_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", ".artifacts", "income_lookup.json"
)


@dataclass
class IncomeClassification:
    category: str
    comments: str  # What to write in column D (description/notes)


class IncomeLookupClassifier:
    """Classifies income transactions using a local lookup table."""

    def __init__(self, lookup_path=None):
        self.lookup_path = lookup_path or DEFAULT_INCOME_LOOKUP_PATH
        self.exact = {}
        self.patterns = []
        self.details_matches = []
        self.load()

    def load(self):
        """Load income lookup table from disk."""
        self.exact = {}
        self.patterns = []
        self.details_matches = []

        if not os.path.exists(self.lookup_path):
            logger.info(
                f"No income lookup table at {self.lookup_path}, starting empty"
            )
            return

        with open(self.lookup_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data.get("exact", []):
            self.exact[entry["action"]] = IncomeClassification(
                category=entry["category"],
                comments=entry.get("comments", ""),
            )

        for entry in data.get("patterns", []):
            self.patterns.append((
                entry["prefix"],
                IncomeClassification(
                    category=entry["category"],
                    comments=entry.get("comments", ""),
                ),
            ))

        for entry in data.get("details_matches", []):
            match_fields = {}
            for field in ("recipient", "sender", "purpose"):
                if entry.get(field):
                    match_fields[field] = entry[field]
            self.details_matches.append((
                match_fields,
                IncomeClassification(
                    category=entry["category"],
                    comments=entry.get("comments", ""),
                ),
            ))

        logger.info(
            f"Loaded income lookup: {len(self.exact)} exact, "
            f"{len(self.patterns)} patterns, "
            f"{len(self.details_matches)} details matches"
        )

    def save(self):
        """Save income lookup table to disk."""
        data = {
            "exact": [
                {"action": action, "category": c.category,
                 "comments": c.comments}
                for action, c in self.exact.items()
            ],
            "patterns": [
                {"prefix": prefix, "category": c.category,
                 "comments": c.comments}
                for prefix, c in self.patterns
            ],
            "details_matches": [
                {**match_fields, "category": c.category,
                 "comments": c.comments}
                for match_fields, c in self.details_matches
            ],
        }
        os.makedirs(os.path.dirname(self.lookup_path), exist_ok=True)
        with open(self.lookup_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved income lookup to {self.lookup_path}")

    def add_exact(self, action, category, comments=""):
        """Add an exact match entry and save."""
        self.exact[action] = IncomeClassification(
            category=category, comments=comments,
        )
        self.save()

    def add_pattern(self, prefix, category, comments=""):
        """Add a prefix pattern entry and save."""
        self.patterns.append((
            prefix,
            IncomeClassification(category=category, comments=comments),
        ))
        self.save()

    def add_details_match(self, match_fields, category, comments=""):
        """Add a details match entry and save.

        Args:
            match_fields: dict with keys from {recipient, sender, purpose}.
                All specified fields must match for the rule to fire.
        """
        self.details_matches.append((
            match_fields,
            IncomeClassification(category=category, comments=comments),
        ))
        self.save()

    def remove_exact(self, action):
        """Remove an exact match entry and save."""
        if action in self.exact:
            del self.exact[action]
            self.save()

    def classify_one(self, action, details=""):
        """Classify a single income transaction.

        Tries action-based matching first (exact, then prefix),
        then falls back to details keyword matching.

        Returns IncomeClassification or None if unknown.
        """
        action = action.strip()

        if action in self.exact:
            return self.exact[action]

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
        """Classify a DataFrame of income transactions.

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
        comments = []
        classified_mask = []

        for _, row in df.iterrows():
            details = str(row.get("details", "")) if "details" in df.columns else ""
            result = self.classify_one(row["action"], details=details)
            if result:
                categories.append(result.category)
                comments.append(result.comments)
                classified_mask.append(True)
            else:
                categories.append("")
                comments.append("")
                classified_mask.append(False)

        classified_mask = pd.Series(classified_mask, index=df.index)

        classified = df[classified_mask].copy()
        if not classified.empty:
            classified["category"] = [
                c for c, m in zip(categories, classified_mask) if m
            ]
            classified["comments"] = [
                c for c, m in zip(comments, classified_mask) if m
            ]

        unclassified = df[~classified_mask].copy()

        count = len(classified)
        total = len(df)
        logger.info(
            f"Income classified {count}/{total} transactions, "
            f"{total - count} unknown"
        )

        return BankClassificationResult(
            classified=classified,
            unclassified=unclassified,
        )
