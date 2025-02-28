"""
Neural network-based classifier for transaction categorization.

This module contains the main classification system that predicts categories
and subcategories for financial transactions.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Concatenate, Input
import logging

from depensage.classifier.feature_extraction import FeatureExtractor

logger = logging.getLogger(__name__)


class ExpenseNeuralClassifier:
    """Neural Network-based expense classifier for financial transactions."""

    def __init__(self, model_dir='models'):
        """
        Initialize the Neural Network-based expense classifier.

        Args:
            model_dir: Directory to save and load models.
        """
        self.model_dir = model_dir
        self.category_model = None
        self.subcategory_models = {}
        self.category_encoder = None
        self.subcategory_encoders = {}
        self.feature_extractor = FeatureExtractor(model_dir)

        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def encode_categories(self, categories, training=False, category_type='main'):
        """
        Encode category labels.

        Args:
            categories: Series of category labels.
            training: Whether this is for training (True) or prediction (False).
            category_type: 'main' for main categories, or the name of the main category for subcategories.

        Returns:
            Encoded categories.
        """
        if training:
            if category_type == 'main':
                self.category_encoder = LabelEncoder()
                encoded = self.category_encoder.fit_transform(categories)

                # Save encoder
                with open(os.path.join(self.model_dir, 'category_encoder.pkl'), 'wb') as f:
                    pickle.dump(self.category_encoder, f)
            else:
                # For subcategories
                encoder = LabelEncoder()
                encoded = encoder.fit_transform(categories)
                self.subcategory_encoders[category_type] = encoder

                # Save encoder
                with open(os.path.join(self.model_dir, f'subcategory_encoder_{category_type}.pkl'), 'wb') as f:
                    pickle.dump(encoder, f)
        else:
            if category_type == 'main':
                if self.category_encoder is None:
                    # Load encoder if not loaded yet
                    try:
                        with open(os.path.join(self.model_dir, 'category_encoder.pkl'), 'rb') as f:
                            self.category_encoder = pickle.load(f)
                    except FileNotFoundError:
                        logger.error("Category encoder not found. Must train model first.")
                        raise

                # Handle new categories that weren't in the training data
                encoded = np.array([
                    self.category_encoder.transform([cat])[0] if cat in self.category_encoder.classes_
                    else -1  # Use -1 for unknown categories
                    for cat in categories
                ])
            else:
                # For subcategories
                if category_type not in self.subcategory_encoders:
                    # Load encoder if not loaded yet
                    try:
                        with open(os.path.join(self.model_dir, f'subcategory_encoder_{category_type}.pkl'), 'rb') as f:
                            self.subcategory_encoders[category_type] = pickle.load(f)
                    except FileNotFoundError:
                        # If encoder for this category doesn't exist, create a new one
                        logger.warning(f"No encoder found for subcategory {category_type}. Creating a new one.")
                        self.subcategory_encoders[category_type] = LabelEncoder()
                        self.subcategory_encoders[category_type].fit(categories)

                # Handle new subcategories
                encoder = self.subcategory_encoders[category_type]
                encoded = np.array([
                    encoder.transform([subcat])[0] if subcat in encoder.classes_
                    else -1  # Use -1 for unknown subcategories
                    for subcat in categories
                ])

        return encoded

    def build_category_model(self, input_shapes, num_categories):
        """
        Build neural network model for main category classification.

        Args:
            input_shapes: Dictionary of input shapes for features.
            num_categories: Number of main categories.

        Returns:
            Compiled Keras model.
        """
        # Business name input and processing
        business_input = Input(shape=(input_shapes['business_features'],), name='business_input')
        business_dense = Dense(64, activation='relu')(business_input)

        # Amount input and processing
        amount_input = Input(shape=(1,), name='amount_input')
        amount_dense = Dense(16, activation='relu')(amount_input)

        # Temporal inputs
        day_of_week_input = Input(shape=(1,), name='day_of_week_input')
        day_of_month_input = Input(shape=(1,), name='day_of_month_input')
        month_input = Input(shape=(1,), name='month_input')

        # Concatenate all features
        concatenated = Concatenate()([
            business_dense,
            amount_dense,
            day_of_week_input,
            day_of_month_input,
            month_input
        ])

        # Dense layers
        dense1 = Dense(128, activation='relu')(concatenated)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(0.2)(dense2)

        # Output layer
        output = Dense(num_categories, activation='softmax')(dropout2)

        # Create model
        model = Model(
            inputs=[
                business_input,
                amount_input,
                day_of_week_input,
                day_of_month_input,
                month_input
            ],
            outputs=output
        )

        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def build_subcategory_model(self, input_shapes, num_subcategories, category_name):
        """
        Build neural network model for subcategory classification.

        Args:
            input_shapes: Dictionary of input shapes for features.
            num_subcategories: Number of subcategories.
            category_name: Name of the main category this model is for.

        Returns:
            Compiled Keras model.
        """
        # Similar architecture as the category model but with different output size
        business_input = Input(shape=(input_shapes['business_features'],), name='business_input')
        business_dense = Dense(32, activation='relu')(business_input)

        amount_input = Input(shape=(1,), name='amount_input')
        amount_dense = Dense(8, activation='relu')(amount_input)

        day_of_week_input = Input(shape=(1,), name='day_of_week_input')
        day_of_month_input = Input(shape=(1,), name='day_of_month_input')
        month_input = Input(shape=(1,), name='month_input')

        concatenated = Concatenate()([
            business_dense,
            amount_dense,
            day_of_week_input,
            day_of_month_input,
            month_input
        ])

        dense1 = Dense(64, activation='relu')(concatenated)
        dropout1 = Dropout(0.2)(dense1)
        dense2 = Dense(32, activation='relu')(dropout1)

        output = Dense(num_subcategories, activation='softmax')(dense2)

        model = Model(
            inputs=[
                business_input,
                amount_input,
                day_of_week_input,
                day_of_month_input,
                month_input
            ],
            outputs=output
        )

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, transactions_df, train_ratio=0.8, epochs=50, batch_size=32):
        """
        Train classification models on historical transaction data.

        Args:
            transactions_df: DataFrame with transaction data including 'business_name',
                             'amount', 'date', 'category', and 'subcategory'.
            train_ratio: Ratio of data to use for training (vs. validation).
            epochs: Number of training epochs.
            batch_size: Batch size for training.

        Returns:
            Dictionary with training history.
        """
        # Check required columns
        required_columns = ['business_name', 'amount', 'date', 'category', 'subcategory']
        missing_columns = [col for col in required_columns if col not in transactions_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Extract features
        features = self.feature_extractor.extract_features(transactions_df, training=True)

        # Encode categories
        category_encoded = self.encode_categories(transactions_df['category'], training=True)

        # Split data for training and validation
        train_idx, val_idx = train_test_split(
            np.arange(len(transactions_df)),
            test_size=1 - train_ratio,
            random_state=42,
            stratify=category_encoded  # Stratify by category to ensure balanced classes
        )

        # Prepare training data
        train_features = {
            'business_input': features['business_features'][train_idx],
            'amount_input': features['amount'][train_idx].reshape(-1, 1),
            'day_of_week_input': features['day_of_week'][train_idx].reshape(-1, 1),
            'day_of_month_input': features['day_of_month'][train_idx].reshape(-1, 1),
            'month_input': features['month'][train_idx].reshape(-1, 1)
        }

        train_category = category_encoded[train_idx]

        # Prepare validation data
        val_features = {
            'business_input': features['business_features'][val_idx],
            'amount_input': features['amount'][val_idx].reshape(-1, 1),
            'day_of_week_input': features['day_of_week'][val_idx].reshape(-1, 1),
            'day_of_month_input': features['day_of_month'][val_idx].reshape(-1, 1),
            'month_input': features['month'][val_idx].reshape(-1, 1)
        }

        val_category = category_encoded[val_idx]

        # Build and train category model
        num_categories = len(np.unique(category_encoded))
        input_shapes = {
            'business_features': features['business_features'].shape[1]
        }

        self.category_model = self.build_category_model(input_shapes, num_categories)

        # Use early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        )

        # Train main category model
        category_history = self.category_model.fit(
            train_features,
            train_category,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_features, val_category),
            callbacks=[early_stopping],
            verbose=1
        )

        # Save main category model
        self.category_model.save(os.path.join(self.model_dir, 'category_model.h5'))

        # Train a separate model for subcategories of each main category
        subcategory_histories = {}
        unique_categories = np.unique(transactions_df['category'])

        for category in unique_categories:
            # Filter transactions for this category
            category_mask = transactions_df['category'] == category
            category_df = transactions_df[category_mask]

            if len(category_df) < 10:  # Skip if too few examples
                logger.warning(f"Not enough data to train subcategory model for {category}")
                continue

            # Extract features for this category's transactions
            cat_features = self.feature_extractor.extract_features(category_df, training=False)

            # Encode subcategories
            subcat_encoded = self.encode_categories(
                category_df['subcategory'],
                training=True,
                category_type=category
            )

            # Split data
            train_idx, val_idx = train_test_split(
                np.arange(len(category_df)),
                test_size=1 - train_ratio,
                random_state=42
            )

            # Prepare training data
            train_features = {
                'business_input': cat_features['business_features'][train_idx],
                'amount_input': cat_features['amount'][train_idx].reshape(-1, 1),
                'day_of_week_input': cat_features['day_of_week'][train_idx].reshape(-1, 1),
                'day_of_month_input': cat_features['day_of_month'][train_idx].reshape(-1, 1),
                'month_input': cat_features['month'][train_idx].reshape(-1, 1)
            }

            train_subcategory = subcat_encoded[train_idx]

            # Prepare validation data
            val_features = {
                'business_input': cat_features['business_features'][val_idx],
                'amount_input': cat_features['amount'][val_idx].reshape(-1, 1),
                'day_of_week_input': cat_features['day_of_week'][val_idx].reshape(-1, 1),
                'day_of_month_input': cat_features['day_of_month'][val_idx].reshape(-1, 1),
                'month_input': cat_features['month'][val_idx].reshape(-1, 1)
            }

            val_subcategory = subcat_encoded[val_idx]

            # Build and train subcategory model
            num_subcategories = len(np.unique(subcat_encoded))
            subcat_model = self.build_subcategory_model(input_shapes, num_subcategories, category)

            # Train subcategory model
            subcat_history = subcat_model.fit(
                train_features,
                train_subcategory,
                epochs=min(epochs, 30),  # Fewer epochs for subcategory models
                batch_size=batch_size,
                validation_data=(val_features, val_subcategory),
                callbacks=[early_stopping],
                verbose=1
            )

            # Save subcategory model
            subcat_model.save(os.path.join(self.model_dir, f'subcategory_model_{category}.h5'))
            self.subcategory_models[category] = subcat_model

            subcategory_histories[category] = subcat_history.history

        # Return training histories
        return {
            'category_history': category_history.history,
            'subcategory_histories': subcategory_histories
        }

    def load_models(self):
        """
        Load all trained models and encoders.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Load main category model
            model_path = os.path.join(self.model_dir, 'category_model.h5')
            if os.path.exists(model_path):
                self.category_model = load_model(model_path)
            else:
                logger.error(f"Category model not found at {model_path}")
                return False

            # Load category encoder
            with open(os.path.join(self.model_dir, 'category_encoder.pkl'), 'rb') as f:
                self.category_encoder = pickle.load(f)

            # Load subcategory models and encoders
            for model_file in os.listdir(self.model_dir):
                if model_file.startswith('subcategory_model_') and model_file.endswith('.h5'):
                    category = model_file[len('subcategory_model_'):-len('.h5')]
                    self.subcategory_models[category] = load_model(
                        os.path.join(self.model_dir, model_file)
                    )

                    # Load corresponding encoder
                    encoder_file = f'subcategory_encoder_{category}.pkl'
                    if os.path.exists(os.path.join(self.model_dir, encoder_file)):
                        with open(os.path.join(self.model_dir, encoder_file), 'rb') as f:
                            self.subcategory_encoders[category] = pickle.load(f)

            logger.info(f"Loaded {len(self.subcategory_models)} subcategory models")
            return True

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False

    def predict(self, transactions_df):
        """
        Predict categories and subcategories for new transactions.

        Args:
            transactions_df: DataFrame with transaction data including 'business_name',
                            'amount', and optionally 'date'.

        Returns:
            DataFrame with added 'predicted_category' and 'predicted_subcategory' columns.
        """
        # Check if models are loaded
        if self.category_model is None:
            success = self.load_models()
            if not success:
                raise RuntimeError("Failed to load classification models")

        # Extract features
        features = self.feature_extractor.extract_features(transactions_df, training=False)

        # Prepare input for model
        model_input = {
            'business_input': features['business_features'],
            'amount_input': features['amount'].reshape(-1, 1),
            'day_of_week_input': features['day_of_week'].reshape(-1, 1),
            'day_of_month_input': features['day_of_month'].reshape(-1, 1),
            'month_input': features['month'].reshape(-1, 1)
        }

        # Predict main categories
        category_probs = self.category_model.predict(model_input)
        category_indices = np.argmax(category_probs, axis=1)

        # Convert indices to category names
        predicted_categories = self.category_encoder.inverse_transform(category_indices)

        # Initialize subcategory predictions
        predicted_subcategories = [''] * len(transactions_df)

        # Predict subcategories for each transaction
        for i, (idx, row) in enumerate(transactions_df.reset_index().iterrows()):
            category = predicted_categories[i]

            # If we have a subcategory model for this category
            if category in self.subcategory_models and category in self.subcategory_encoders:
                # Extract features for this transaction
                single_model_input = {
                    'business_input': features['business_features'][i:i + 1],
                    'amount_input': features['amount'][i:i + 1].reshape(-1, 1),
                    'day_of_week_input': features['day_of_week'][i:i + 1].reshape(-1, 1),
                    'day_of_month_input': features['day_of_month'][i:i + 1].reshape(-1, 1),
                    'month_input': features['month'][i:i + 1].reshape(-1, 1)
                }

                # Predict subcategory
                subcategory_probs = self.subcategory_models[category].predict(single_model_input)
                subcategory_idx = np.argmax(subcategory_probs, axis=1)[0]

                # Convert index to subcategory name
                encoder = self.subcategory_encoders[category]
                predicted_subcategories[i] = encoder.inverse_transform([subcategory_idx])[0]

        # Add predictions to the DataFrame
        result_df = transactions_df.copy()
        result_df['predicted_category'] = predicted_categories
        result_df['predicted_subcategory'] = predicted_subcategories

        return result_df
