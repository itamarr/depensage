"""
Lookup-based transaction classifier.

Classifies transactions by matching merchant names against a lookup table
of known merchants. Supports exact matches and prefix patterns.
"""

import json
import logging
import os
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_LOOKUP_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", ".artifacts", "lookup.json"
)


@dataclass
class Classification:
    category: str
    subcategory: str


@dataclass
class ClassificationResult:
    classified: pd.DataFrame
    unclassified: pd.DataFrame


class LookupClassifier:
    """Classifies transactions using a local lookup table."""

    def __init__(self, lookup_path=None):
        self.lookup_path = lookup_path or DEFAULT_LOOKUP_PATH
        self.exact: dict[str, Classification] = {}
        self.patterns: list[tuple[str, Classification]] = []
        self.load()

    def load(self):
        """Load lookup table from disk."""
        self.exact = {}
        self.patterns = []

        if not os.path.exists(self.lookup_path):
            logger.info(f"No lookup table at {self.lookup_path}, starting empty")
            return

        with open(self.lookup_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for name, entry in data.get("exact", {}).items():
            self.exact[name] = Classification(
                category=entry["category"],
                subcategory=entry.get("subcategory", ""),
            )

        for entry in data.get("patterns", []):
            self.patterns.append((
                entry["prefix"],
                Classification(
                    category=entry["category"],
                    subcategory=entry.get("subcategory", ""),
                ),
            ))

        logger.info(
            f"Loaded lookup table: {len(self.exact)} exact, {len(self.patterns)} patterns"
        )

    def save(self):
        """Save lookup table to disk."""
        os.makedirs(os.path.dirname(self.lookup_path), exist_ok=True)

        data = {
            "exact": {
                name: {"category": c.category, "subcategory": c.subcategory}
                for name, c in self.exact.items()
            },
            "patterns": [
                {"prefix": prefix, "category": c.category, "subcategory": c.subcategory}
                for prefix, c in self.patterns
            ],
        }

        with open(self.lookup_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved lookup table to {self.lookup_path}")

    def match(self, merchant_name: str) -> Classification | None:
        """Look up a single merchant name. Returns None if unknown or empty."""
        if not merchant_name or not isinstance(merchant_name, str):
            return None

        # Exact match first
        if merchant_name in self.exact:
            return self.exact[merchant_name]

        # Pattern match (case-insensitive)
        name_lower = merchant_name.lower()
        for prefix, classification in self.patterns:
            if name_lower.startswith(prefix.lower()):
                return classification

        return None

    def classify(self, transactions: pd.DataFrame) -> ClassificationResult:
        """
        Classify a DataFrame of transactions.

        Expects a 'business_name' column. Adds 'category' and 'subcategory'
        columns to classified rows.

        Returns a ClassificationResult with classified and unclassified DataFrames.
        """
        categories = []
        subcategories = []
        matched_mask = []

        for _, row in transactions.iterrows():
            result = self.match(row["business_name"])
            if result:
                categories.append(result.category)
                subcategories.append(result.subcategory)
                matched_mask.append(True)
            else:
                categories.append("")
                subcategories.append("")
                matched_mask.append(False)

        matched_series = pd.Series(matched_mask, index=transactions.index)

        classified = transactions[matched_series].copy()
        classified["category"] = [c for c, m in zip(categories, matched_mask) if m]
        classified["subcategory"] = [s for s, m in zip(subcategories, matched_mask) if m]

        unclassified = transactions[~matched_series].copy()

        logger.info(
            f"Classified {len(classified)}/{len(transactions)} transactions, "
            f"{len(unclassified)} unknown"
        )

        return ClassificationResult(classified=classified, unclassified=unclassified)

    def add_exact(self, merchant_name: str, category: str, subcategory: str = ""):
        """Add an exact match entry and save."""
        self.exact[merchant_name] = Classification(category=category, subcategory=subcategory)
        self.save()

    def add_pattern(self, prefix: str, category: str, subcategory: str = ""):
        """Add a prefix pattern entry and save."""
        self.patterns.append((prefix, Classification(category=category, subcategory=subcategory)))
        self.save()
