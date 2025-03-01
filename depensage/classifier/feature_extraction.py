"""
Feature extraction module for transaction classification.

This module handles the preprocessing and feature extraction from transaction data
for use in the neural network classifier.
"""

import numpy as np
import pandas as pd
import re
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts and processes features from transaction data for classification.
    """

    def __init__(self, model_dir='models'):
        """
        Initialize the feature extractor.

        Args:
            model_dir: Directory to save and load models.
        """
        self.model_dir = model_dir
        self.business_vectorizer = None
        self.amount_scaler = None

        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def preprocess_business_name(self, name):
        """
        Preprocess business name for better feature extraction.

        Args:
            name: Business name string.

        Returns:
            Processed business name.
        """
        if not isinstance(name, str):
            return ""

        # Convert to lowercase
        name = name.lower()

        # Remove quotes and special characters
        name = re.sub(r'["\'\(\)\[\]\{\}]', ' ', name)

        # Replace multiple spaces with single space
        name = re.sub(r'\s+', ' ', name)

        return name.strip()

    def extract_features(self, transactions_df, training=False):
        """
        Extract features from transactions dataframe.

        Args:
            transactions_df: DataFrame with transactions.
            training: Whether this is for training (True) or prediction (False).

        Returns:
            Dictionary of features.
        """
        # Preprocess business names
        business_names = transactions_df['business_name'].fillna('').apply(self.preprocess_business_name)

        # Extract amount features
        amounts = transactions_df['amount'].values.reshape(-1, 1)

        # Extract temporal features
        if 'date' in transactions_df.columns:
            transactions_df['day_of_week'] = transactions_df['date'].dt.dayofweek
            transactions_df['day_of_month'] = transactions_df['date'].dt.day
            transactions_df['month'] = transactions_df['date'].dt.month

            day_of_week = transactions_df['day_of_week'].values
            day_of_month = transactions_df['day_of_month'].values
            month = transactions_df['month'].values
        else:
            # Default values if date is not available
            day_of_week = np.zeros(len(transactions_df))
            day_of_month = np.zeros(len(transactions_df))
            month = np.zeros(len(transactions_df))

        # TF-IDF Vectorization for business names
        if training:
            self.business_vectorizer = TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(2, 5),
                min_df=2,
                max_features=1000
            )
            business_features = self.business_vectorizer.fit_transform(business_names).toarray()

            # Scale amount
            self.amount_scaler = StandardScaler()
            amount_scaled = self.amount_scaler.fit_transform(amounts)

            # Save vectorizer and scaler
            with open(os.path.join(self.model_dir, 'business_vectorizer.pkl'), 'wb') as f:
                pickle.dump(self.business_vectorizer, f)

            with open(os.path.join(self.model_dir, 'amount_scaler.pkl'), 'wb') as f:
                pickle.dump(self.amount_scaler, f)
        else:
            if self.business_vectorizer is None:
                # Load vectorizer if not loaded yet
                try:
                    with open(os.path.join(self.model_dir, 'business_vectorizer.pkl'), 'rb') as f:
                        self.business_vectorizer = pickle.load(f)
                except FileNotFoundError:
                    logger.error("Business vectorizer not found. Must train model first.")
                    raise

            if self.amount_scaler is None:
                # Load scaler if not loaded yet
                try:
                    with open(os.path.join(self.model_dir, 'amount_scaler.pkl'), 'rb') as f:
                        self.amount_scaler = pickle.load(f)
                except FileNotFoundError:
                    logger.error("Amount scaler not found. Must train model first.")
                    raise

            business_features = self.business_vectorizer.transform(business_names).toarray()
            amount_scaled = self.amount_scaler.transform(amounts)

        # Create feature dictionary
        features = {
            'business_features': business_features,
            'amount': amount_scaled.flatten(),
            'day_of_week': day_of_week,
            'day_of_month': day_of_month,
            'month': month
        }

        return features
