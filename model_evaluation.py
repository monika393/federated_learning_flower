"""
Model Evaluation Module

This module contains classes responsible for training centralized models
and evaluating federated learning models for performance comparison.
"""

import numpy as np
from typing import Tuple, TYPE_CHECKING
from sklearn.metrics import mean_squared_error
from model_management import ModelFactory

if TYPE_CHECKING:
    from federated_client import SklearnClient


class CentralizedTrainer:
    """
    Handles centralized model training for comparison purposes.
    
    This class manages the training of centralized models that can be
    used as benchmarks for comparing federated learning performance.
    It provides a standardized approach to centralized training.
    """
    
    @staticmethod
    def train_gradient_boosting(X: np.ndarray, y: np.ndarray) -> Tuple[object, float]:
        """
        Train a centralized GradientBoostingRegressor model.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Tuple of (trained model, MSE score)
        """
        print("Training centralized model for comparison...")
        model = ModelFactory.create_gradient_boosting_regressor()
        model.fit(X, y)
        preds = model.predict(X)
        mse = mean_squared_error(y, preds)
        print(f"Centralized Gradient Boosting Regressor MSE: {mse:.4f}")
        return model, mse


class FederatedModelEvaluator:
    """
    Evaluates federated models on the full dataset.
    
    This class provides methods for evaluating federated learning
    models by aggregating client models and testing them on the
    complete dataset. It enables performance comparison between
    federated and centralized approaches.
    """
    
    @staticmethod
    def evaluate_aggregated_model(client1: 'SklearnClient', client2: 'SklearnClient', 
                                X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate the aggregated federated model on the full dataset.
        
        Args:
            client1: First trained client
            client2: Second trained client
            X: Feature matrix for evaluation
            y: Target values for evaluation
            
        Returns:
            MSE score of the federated model
        """
        final_coef = (client1.model.coef_ + client2.model.coef_) / 2
        final_intercept = (client1.model.intercept_ + client2.model.intercept_) / 2
        federated_preds = X.dot(final_coef) + final_intercept
        federated_mse = mean_squared_error(y, federated_preds)
        print(f"Federated SGDRegressor approximated final MSE: {federated_mse:.4f}")
        return federated_mse 