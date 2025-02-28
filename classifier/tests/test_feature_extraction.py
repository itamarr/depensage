"""
Unit tests for the feature extraction module.
"""

import unittest
import os
import shutil
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from depensage.classifier.feature_extraction import FeatureExtractor


class TestFeatureExtractor(unittest.TestCase):
    """Test cases for the FeatureExtractor class."""

    def setUp(self):
        """Set up test environment before each test method."""
        # Create a temporary directory for model files
        self.test_model_dir = tempfile.mkdtemp()
        self.extractor = FeatureExtractor(model_dir=self.test_model_dir)

        # Create sample transaction data
        self.sample_data = pd.DataFrame({
            'date': [
                datetime.now() - timedelta(days=i)
                for i in range(10)
            ],
            'business_name': [
                'Supermarket A' if i % 2 == 0 else 'Restaurant B'
                for i in range(10)
            ],
            'amount': [
                100.0 if i % 2 == 0 else 50.0
                for i in range(10)
            ]
        })

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_model_dir)

    def test_preprocess_business_name(self):
        """Test business name preprocessing."""
        # Test lowercase conversion
        self.assertEqual(
            self.extractor.preprocess_business_name('SUPERMARKET'),
            'supermarket'
        )

        # Test special character removal
        self.assertEqual(
            self.extractor.preprocess_business_name('Restaurant (City Center)'),
            'restaurant city center'
        )

        # Test empty input
        self.assertEqual(
            self.extractor.preprocess_business_name(''),
            ''
        )

        # Test non-string input
        self.assertEqual(
            self.extractor.preprocess_business_name(123),
            ''
        )

    def test_extract_features_training_mode(self):
        """Test feature extraction in training mode."""
        features = self.extractor.extract_features(self.sample_data, training=True)

        # Check that we have all the expected features
        self.assertIn('business_features', features)
        self.assertIn('amount', features)
        self.assertIn('day_of_week', features)
        self.assertIn('day_of_month', features)
        self.assertIn('month', features)

        # Check that they have the right shapes
        self.assertEqual(len(features['amount']), 10)
        self.assertEqual(len(features['day_of_week']), 10)

        # Check that vectorizer and scaler were created and saved
        self.assertIsNotNone(self.extractor.business_vectorizer)
        self.assertIsNotNone(self.extractor.amount_scaler)
        self.assertTrue(os.path.exists(os.path.join(self.test_model_dir, 'business_vectorizer.pkl')))
        self.assertTrue(os.path.exists(os.path.join(self.test_model_dir, 'amount_scaler.pkl')))

    def test_extract_features_prediction_mode(self):
        """Test feature extraction in prediction mode."""
        # First train to create the vectorizer and scaler
        self.extractor.extract_features(self.sample_data, training=True)

        # Create a new extractor to test loading
        new_extractor = FeatureExtractor(model_dir=self.test_model_dir)

        # Now test prediction mode
        features = new_extractor.extract_features(self.sample_data, training=False)

        # Check that features are still correct
        self.assertIn('business_features', features)
        self.assertIn('amount', features)
        self.assertEqual(len(features['amount']), 10)

        # Check that vectorizer and scaler were loaded
        self.assertIsNotNone(new_extractor.business_vectorizer)
        self.assertIsNotNone(new_extractor.amount_scaler)

    def test_extract_features_no_date(self):
        """Test feature extraction with no date column."""
        # Create data without date column
        no_date_data = pd.DataFrame({
            'business_name': ['Supermarket A', 'Restaurant B'],
            'amount': [100.0, 50.0]
        })

        # Extract features
        features = self.extractor.extract_features(no_date_data, training=True)

        # Check that temporal features are zeros
        self.assertTrue(np.all(features['day_of_week'] == 0))
        self.assertTrue(np.all(features['day_of_month'] == 0))
        self.assertTrue(np.all(features['month'] == 0))

    @patch('sklearn.feature_extraction.text.TfidfVectorizer')
    def test_tfidf_vectorization(self, mock_vectorizer):
        """Test TF-IDF vectorization of business names."""
        # Mock vectorizer
        mock_vectorizer_instance = mock_vectorizer.return_value
        mock_vectorizer_instance.fit_transform.return_value = np.array([[1, 0], [0, 1]])
        mock_vectorizer_instance.transform.return_value = np.array([[1, 0], [0, 1]])

        # Replace the real vectorizer with the mock
        with patch('sklearn.feature_extraction.text.TfidfVectorizer', return_value=mock_vectorizer_instance):
            # Extract features in training mode
            self.extractor.extract_features(self.sample_data.iloc[:2], training=True)

            # Check that fit_transform was called
            mock_vectorizer_instance.fit_transform.assert_called_once()


if __name__ == '__main__':
    unittest.main()