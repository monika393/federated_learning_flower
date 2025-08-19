"""
Federated Learning Client Module

This module contains the core federated learning client implementation
and federated averaging strategy with loss tracking capabilities.
"""

import flwr as fl
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sklearn.metrics import mean_squared_error, r2_score
from flwr.server.strategy import FedAvg
from model_management import ModelFactory, ParameterManager
from training_tracking import LossTracker


class SklearnClient(fl.client.NumPyClient):
    """
    A Flower client implementation using scikit-learn SGDRegressor.
    
    This client maintains a local model and dataset, implementing the
    federated learning protocol for parameter sharing and local training.
    It handles local model training, parameter extraction, and evaluation
    while maintaining privacy by keeping data local.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, client_id: int):
        """
        Initialize the client with local data and model.
        
        Args:
            X: Feature matrix for local training data
            y: Target values for local training data
            client_id: Unique identifier for this client
        """
        self.model = ModelFactory.create_sgd_regressor()
        self.X = X
        self.y = y
        self.client_id = client_id
        self.loss_tracker = LossTracker()

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """
        Return current model parameters for federated aggregation.
        
        Args:
            config: Configuration dictionary (unused in this implementation)
            
        Returns:
            List containing model coefficients and intercept
        """
        return ParameterManager.extract_parameters(self.model)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Update model parameters and fit on local data.
        
        Args:
            parameters: Model parameters from server aggregation
            config: Configuration dictionary (unused in this implementation)
            
        Returns:
            Tuple of (updated parameters, number of samples, metrics)
        """
        ParameterManager.set_parameters(self.model, parameters)
        self.model.fit(self.X, self.y)
        
        # Calculate training metrics
        preds = self.model.predict(self.X)
        mse = mean_squared_error(self.y, preds)
        r2 = r2_score(self.y, preds)
        
        # Record loss
        self.loss_tracker.record_client_loss(mse)
        
        print(f"Client {self.client_id} training - MSE: {mse:.4f}, R²: {r2:.4f}")
        
        return self.get_parameters(config), len(self.X), {
            "mse": mse,
            "r2": r2,
            "num_samples": len(self.X)
        }

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluate model performance on local data.
        
        Args:
            parameters: Model parameters to evaluate
            config: Configuration dictionary (unused in this implementation)
            
        Returns:
            Tuple of (loss, number of samples, metrics)
        """
        ParameterManager.set_parameters(self.model, parameters)
        preds = self.model.predict(self.X)
        mse = mean_squared_error(self.y, preds)
        r2 = r2_score(self.y, preds)
        
        # Record loss
        self.loss_tracker.record_client_loss(mse)
        
        print(f"Client {self.client_id} evaluation - MSE: {mse:.4f}, R²: {r2:.4f}")
        
        return mse, len(self.X), {
            "accuracy": r2,  # Using R² as accuracy metric for regression
            "mse": mse,
            "r2": r2,
            "num_samples": len(self.X)
        }


class FedAvgWithLossTracking(FedAvg):
    """
    Federated Averaging strategy with loss tracking capabilities.
    
    Extends the standard FedAvg strategy to track and log aggregated
    evaluation losses across training rounds. This class provides
    enhanced monitoring capabilities for federated learning experiments
    by recording global loss progression.
    """
    
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """
        Aggregate evaluation results and track global loss.
        
        Args:
            rnd: Current round number
            results: List of client evaluation results
            failures: List of client failures
            
        Returns:
            Aggregated loss value or None if aggregation fails
        """
        aggregated_loss = super().aggregate_evaluate(rnd, results, failures)
        if aggregated_loss is not None:
            global_losses.append(aggregated_loss)
            print(f"🌐 Round {rnd} aggregated server loss: {aggregated_loss:.4f}")
        return aggregated_loss 