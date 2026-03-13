"""
Lookup-based classifier for bank account transactions.

Classifies bank debits and credits by matching the action name (הפעולה)
against a lookup table. Supports exact matches and prefix patterns,
same as the CC lookup but keyed on action name instead of merchant name.
"""

import json
import logging
import os
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_BANK_LOOKUP_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", ".artifacts", "bank_lookup.json"
)

DEFAULT_INCOME_LOOKUP_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", ".artifacts", "income_lookup.json"
)


@dataclass
class BankClassification:
    category: str
    subcategory: str
    business_name: str  # What to write in column B (for expenses)


@dataclass
class IncomeClassification:
    category: str
    comments: str  # What to write in column D (description/notes)


@dataclass
class BankClassificationResult:
    classified: pd.DataFrame
    unclassified: pd.DataFrame


class BankLookupClassifier:
    """Classifies bank transactions using a local lookup table."""

    def __init__(self, lookup_path=None):
        self.lookup_path = lookup_path or DEFAULT_BANK_LOOKUP_PATH
        self.exact = {}
        self.patterns = []
        self.load()

    def load(self):
        """Load lookup table from disk."""
        self.exact = {}
        self.patterns = []

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
                business_name=entry.get("business_name", ""),
            )

        for entry in data.get("patterns", []):
            self.patterns.append((
                entry["prefix"],
                BankClassification(
                    category=entry["category"],
                    subcategory=entry.get("subcategory", ""),
                    business_name=entry.get("business_name", ""),
                ),
            ))

        logger.info(
            f"Loaded bank lookup: {len(self.exact)} exact, "
            f"{len(self.patterns)} patterns"
        )

    def classify_one(self, action):
        """Classify a single action name.

        Returns BankClassification or None if unknown.
        """
        action = action.strip()

        # Exact match
        if action in self.exact:
            return self.exact[action]

        # Prefix match
        for prefix, classification in self.patterns:
            if action.startswith(prefix):
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
        business_names = []
        classified_mask = []

        for _, row in df.iterrows():
            result = self.classify_one(row["action"])
            if result:
                categories.append(result.category)
                subcategories.append(result.subcategory)
                business_names.append(result.business_name)
                classified_mask.append(True)
            else:
                categories.append("")
                subcategories.append("")
                business_names.append("")
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
            classified["business_name"] = [
                b for b, m in zip(business_names, classified_mask) if m
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


class IncomeLookupClassifier:
    """Classifies income transactions using a local lookup table."""

    def __init__(self, lookup_path=None):
        self.lookup_path = lookup_path or DEFAULT_INCOME_LOOKUP_PATH
        self.exact = {}
        self.patterns = []
        self.load()

    def load(self):
        """Load income lookup table from disk."""
        self.exact = {}
        self.patterns = []

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

        logger.info(
            f"Loaded income lookup: {len(self.exact)} exact, "
            f"{len(self.patterns)} patterns"
        )

    def classify_one(self, action):
        """Classify a single action name.

        Returns IncomeClassification or None if unknown.
        """
        action = action.strip()

        if action in self.exact:
            return self.exact[action]

        for prefix, classification in self.patterns:
            if action.startswith(prefix):
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
            result = self.classify_one(row["action"])
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
