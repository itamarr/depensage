# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DepenSage (dépense + sage) is a Python expense tracking tool that uses neural networks to classify credit card transactions and update Google Sheets spreadsheets. It supports Hebrew language spreadsheets.

## Commands

```bash
# Install for development
pip install -e .

# Run all tests
python -m unittest discover

# Run a single test file
python -m unittest depensage/config/tests/test_settings.py

# Run a specific test
python -m unittest depensage.config.tests.test_settings.TestLoadSettings.test_load_settings_success

# Lint and format
flake8 depensage/
black depensage/
```

## Architecture

**CLI entry point** (`main.py`) exposes three commands via argparse: `configure`, `train`, `process`.

**Processing pipeline** (`engine/expense_processor.py` - `ExpenseProcessor`):
1. `StatementParser` reads CSV credit card statements (utf-8-sig encoding for Hebrew), merges multiple statements, filters by date
2. `ExpenseNeuralClassifier` predicts category + subcategory using a hierarchical approach: one main-category model, then per-category subcategory models (TensorFlow/Keras)
3. `SheetHandler` authenticates via Google service account, creates monthly sheets, and writes classified transactions

**Feature extraction** (`classifier/feature_extraction.py` - `FeatureExtractor`): TF-IDF on business names (char n-grams 2-5), StandardScaler on amounts, temporal features from dates.

**Configuration** (`config/settings.py`): Module-level singleton cache. Config lives at `~/.depensage/config.json`. Required keys: `spreadsheet_id`, `credentials_file`.

## Key Conventions

- Tests use `unittest` (not pytest classes), located in `<module>/tests/` subdirectories
- External dependencies (Google Sheets API, Keras models) are mocked in tests
- Transactions flow as pandas DataFrames with columns: `date`, `business_name`, `amount`, `category`, `subcategory`
- The sheets module maps to a specific spreadsheet layout: columns A-F (metadata), G (date), with fixed positions for subcategory/amount/category
- Settings module uses `Settings` class in `main.py` but `load_settings()` function pattern internally — these are being consolidated
