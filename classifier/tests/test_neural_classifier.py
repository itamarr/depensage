"""
Unit tests for the neural classifier module.
"""

import unittest
import os
import shutil
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
from unittest.mock import patch, MagicMock

from depensage.classifier.neural_classifier import ExpenseNeuralClassifier


class TestExpenseNeuralClassifier(unittest.TestCase):
    """Test cases for the ExpenseNeuralClassifier class."""

    def setUp(self):
        """Set up test environment before each test method."""
        # Create a temporary directory for model files
        self.test_model_dir = tempfile.mkdtemp()
        self.classifier = ExpenseNeuralClassifier(model_dir=self.test_model_dir)

        # Create sample transaction data
        self.sample_data = pd.DataFrame({
            'date': [
                datetime.now() - timedelta(days=i)
                for i in range(50)
            ],
            'business_name': [
                'Supermarket A' if i % 5 == 0 else
                'Restaurant B' if i % 5 == 1 else
                'Gas Station C' if i % 5 == 2 else
                'Online Store D' if i % 5 == 3 else
                'Pharmacy E'
                for i in range(50)
            ],
            'amount': [
                100.0 if i % 5 == 0 else
                50.0 if i % 5 == 1 else
                30.0 if i % 5 == 2 else
                75.0 if i % 5 == 3 else
                25.0
                for i in range(50)
            ],
            'category': [
                'Groceries' if i % 5 == 0 else
                'Dining' if i % 5 == 1 else
                'Transportation' if i % 5 == 2 else
                'Shopping' if i % 5 == 3 else
                'Health'
                for i in range(50)
            ],
            'subcategory': [
                'Food' if i % 5 == 0 else
                'Restaurant' if i % 5 == 1 else
                'Fuel' if i % 5 == 2 else
                'Clothing' if i % 5 == 3 else
                'Medicine'
                for i in range(50)
            ]
        })

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_model_dir)

    def test_encode_categories(self):
        """Test category encoding."""
        categories = ['Groceries', 'Dining', 'Transportation', 'Dining']

        # Test training mode
        encoded = self.classifier.encode_categories(categories, training=True)
        self.assertEqual(len(encoded), 4)
        self.assertEqual(len(np.unique(encoded)), 3)  # 3 unique categories

        # Test prediction mode
        encoded_pred = self.classifier.encode_categories(categories, training=False)
        self.assertEqual(len(encoded_pred), 4)

        # Test subcategory encoding
        subcategories = ['Food', 'Drinks', 'Dessert']
        encoded_sub = self.classifier.encode_categories(
            subcategories, training=True, category_type='Dining'
        )
        self.assertEqual(len(encoded_sub), 3)

    @patch('tensorflow.keras.models.Model')
    def test_build_category_model(self, mock_model):
        """Test model building."""
        # Mock Keras model
        mock_model.return_value.compile.return_value = None

        input_shapes = {'business_features': 100}
        num_categories = 5

        model = self.classifier.build_category_model(input_shapes, num_categories)

        # Check that the model was built
        mock_model.assert_called_once()

    @patch('tensorflow.keras.models.Model')
    def test_build_subcategory_model(self, mock_model):
        """Test subcategory model building."""
        # Mock Keras model
        mock_model.return_value.compile.return_value = None

        input_shapes = {'business_features': 100}
        num_subcategories = 3
        category_name = 'Dining'

        model = self.classifier.build_subcategory_model(
            input_shapes, num_subcategories, category_name
        )

        # Check that the model was built
        mock_model.assert_called_once()

    @patch('tensorflow.keras.models.Model.fit')
    @patch('tensorflow.keras.models.Model.save')
    def test_train(self, mock_save, mock_fit):
        """Test model training."""
        # Mock model training and saving
        mock_history = MagicMock()
        mock_history.history = {'accuracy': [0.7, 0.8, 0.9], 'val_accuracy': [0.6, 0.7, 0.8]}
        mock_fit.return_value = mock_history

        # Mock feature extraction
        with patch.object(self.classifier.feature_extractor, 'extract_features') as mock_extract:
            # Mock feature extraction to return appropriate shapes
            mock_extract.return_value = {
                'business_features': np.random.rand(50, 100),
                'amount': np.random.rand(50),
                'day_of_week': np.random.randint(0, 7, 50),
                'day_of_month': np.random.randint(1, 31, 50),
                'month': np.random.randint(1, 13, 50)
            }

            # Train the model
            history = self.classifier.train(self.sample_data, epochs=3, batch_size=16)

            # Check that training was called
            self.assertEqual(mock_fit.call_count, 1)  # Expecting main category model to be trained

            # Check that history was returned
            self.assertIn('category_history', history)

    @patch('tensorflow.keras.models.load_model')
    def test_load_models(self, mock_load):
        """Test loading trained models."""
        # Mock file operations
        with patch('os.path.exists', return_value=True), \
                patch('os.listdir', return_value=['category_model.h5', 'subcategory_model_Dining.h5']), \
                patch('pickle.load', return_value=MagicMock()):
            # Mock loading models
            mock_load.return_value = MagicMock()

            # Test loading
            result = self.classifier.load_models()

            # Check result
            self.assertTrue(result)
            self.assertEqual(mock_load.call_count, 2)  # Main model + one subcategory model

    @patch('tensorflow.keras.models.Model.predict')
    def test_predict(self, mock_predict):
        """Test transaction prediction."""
        # Create a small test dataset
        test_data = pd.DataFrame({
            'date': [datetime.now(), datetime.now() - timedelta(days=1)],
            'business_name': ['Supermarket X', 'Restaurant Y'],
            'amount': [75.0, 30.0]
        })

        # Mock model loading and prediction
        with patch.object(self.classifier, 'load_models', return_value=True), \
                patch.object(self.classifier.feature_extractor, 'extract_features') as mock_extract:
            # Mock feature extraction
            mock_extract.return_value = {
                'business_features': np.random.rand(2, 100),
                'amount': np.random.rand(2),
                'day_of_week': np.random.randint(0, 7, 2),
                'day_of_month': np.random.randint(1, 31, 2),
                'month': np.random.randint(1, 13, 2)
            }

            # Mock category prediction
            self.classifier.category_model = MagicMock()
            mock_predict.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])

            # Mock encoder
            self.classifier.category_encoder = MagicMock()
            self.classifier.category_encoder.inverse_transform.return_value = ['Groceries', 'Dining']

            # No subcategory models for this test
            self.classifier.subcategory_models = {}

            # Predict
            result_df = self.classifier.predict(test_data)

            # Check results
            self.assertEqual(len(result_df), 2)
            self.assertTrue('predicted_category' in result_df.columns)
            self.assertTrue('predicted_subcategory' in result_df.columns)


if __name__ == '__main__':
    unittest.main()
