"""
Lookup table CRUD endpoints for CC, bank, and income classifiers.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from depensage.classifier.cc_lookup import LookupClassifier, Classification
from depensage.classifier.bank_lookup import BankLookupClassifier, BankClassification
from depensage.classifier.income_lookup import IncomeLookupClassifier, IncomeClassification
from depensage.web.auth import require_auth

router = APIRouter(
    prefix="/api/lookups", tags=["lookups"],
    dependencies=[Depends(require_auth)],
)


class ExactEntry(BaseModel):
    key: str
    category: str
    subcategory: str = ""


class PatternEntry(BaseModel):
    prefix: str
    category: str
    subcategory: str = ""


class DetailsMatchEntry(BaseModel):
    match_fields: dict[str, str]
    category: str
    subcategory: str = ""


def _get_classifier(lookup_type: str):
    if lookup_type == "cc":
        return LookupClassifier()
    elif lookup_type == "bank":
        return BankLookupClassifier()
    elif lookup_type == "income":
        return IncomeLookupClassifier()
    raise HTTPException(status_code=400, detail=f"Unknown type: {lookup_type}")


@router.get("/{lookup_type}")
async def get_all(lookup_type: str):
    """Get all entries for a classifier type."""
    cls = _get_classifier(lookup_type)

    exact = []
    if lookup_type == "cc":
        for name, c in cls.exact.items():
            exact.append({"key": name, "category": c.category, "subcategory": c.subcategory})
    else:
        for name, c in cls.exact.items():
            sub = getattr(c, "subcategory", "") or getattr(c, "comments", "")
            exact.append({"key": name, "category": c.category, "subcategory": sub})

    patterns = []
    for prefix, c in cls.patterns:
        sub = getattr(c, "subcategory", "") or getattr(c, "comments", "")
        patterns.append({"prefix": prefix, "category": c.category, "subcategory": sub})

    details_matches = []
    if hasattr(cls, "details_matches"):
        for fields, c in cls.details_matches:
            sub = getattr(c, "subcategory", "") or getattr(c, "comments", "")
            details_matches.append({
                "match_fields": fields, "category": c.category, "subcategory": sub,
            })

    return {
        "type": lookup_type,
        "exact": exact,
        "patterns": patterns,
        "details_matches": details_matches,
    }


@router.post("/{lookup_type}/exact")
async def add_exact(lookup_type: str, entry: ExactEntry):
    """Add an exact match entry."""
    cls = _get_classifier(lookup_type)
    if lookup_type == "income":
        cls.add_exact(entry.key, entry.category, entry.subcategory)
    else:
        cls.add_exact(entry.key, entry.category, entry.subcategory)
    return {"status": "added", "key": entry.key}


@router.put("/{lookup_type}/exact/{key}")
async def update_exact(lookup_type: str, key: str, entry: ExactEntry):
    """Update an exact match entry."""
    cls = _get_classifier(lookup_type)
    if key in cls.exact:
        cls.remove_exact(key)
    if lookup_type == "income":
        cls.add_exact(entry.key, entry.category, entry.subcategory)
    else:
        cls.add_exact(entry.key, entry.category, entry.subcategory)
    return {"status": "updated", "key": entry.key}


@router.delete("/{lookup_type}/exact/{key}")
async def delete_exact(lookup_type: str, key: str):
    """Remove an exact match entry."""
    cls = _get_classifier(lookup_type)
    if key not in cls.exact:
        raise HTTPException(status_code=404, detail=f"Key '{key}' not found")
    cls.remove_exact(key)
    cls.save()
    return {"status": "deleted", "key": key}


@router.post("/{lookup_type}/pattern")
async def add_pattern(lookup_type: str, entry: PatternEntry):
    """Add a prefix pattern."""
    cls = _get_classifier(lookup_type)
    if lookup_type == "income":
        cls.add_pattern(entry.prefix, entry.category, entry.subcategory)
    else:
        cls.add_pattern(entry.prefix, entry.category, entry.subcategory)
    return {"status": "added", "prefix": entry.prefix}


@router.delete("/{lookup_type}/pattern/{index}")
async def delete_pattern(lookup_type: str, index: int):
    """Remove a pattern by index."""
    cls = _get_classifier(lookup_type)
    if index < 0 or index >= len(cls.patterns):
        raise HTTPException(status_code=404, detail=f"Pattern index {index} out of range")
    cls.patterns.pop(index)
    cls.save()
    return {"status": "deleted", "index": index}
