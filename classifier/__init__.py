"""
DepenSage Classifier Subpackage

This subpackage contains the neural network-based classification system
for categorizing financial transactions.
"""

from depensage.classifier.neural_classifier import ExpenseNeuralClassifier
from depensage.classifier.feature_extraction import FeatureExtractor

__all__ = ['ExpenseNeuralClassifier', 'FeatureExtractor']
