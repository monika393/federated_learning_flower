"""
Model Management Module

This module contains classes responsible for creating, configuring, and managing
machine learning models and their parameters for federated learning scenarios.
"""

import numpy as np
from typing import List
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor


class ModelFactory:
    """
    Creates and configures machine learning models.
    
    This class serves as a factory for creating different types of
    machine learning models with appropriate configurations. It
    centralizes model creation logic and ensures consistent model
    initialization across the application.
    """
    
    @staticmethod
    def create_sgd_regressor() -> SGDRegressor:
        """
        Create a configured SGDRegressor for federated learning.
        
        Returns:
            Configured SGDRegressor instance
        """
        return SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
    
    @staticmethod
    def create_gradient_boosting_regressor() -> GradientBoostingRegressor:
        """
        Create a configured GradientBoostingRegressor for centralized comparison.
        
        Returns:
            Configured GradientBoostingRegressor instance
        """
        return GradientBoostingRegressor(random_state=42)


class ParameterManager:
    """
    Manages model parameters for federated learning.
    
    This class handles the extraction, manipulation, and aggregation
    of model parameters during federated learning. It provides
    utilities for parameter averaging and parameter transfer between
    clients and server.
    """
    
    @staticmethod
    def extract_parameters(model: SGDRegressor) -> List[np.ndarray]:
        """
        Extract parameters from a scikit-learn model.
        
        Args:
            model: The model to extract parameters from
            
        Returns:
            List containing model coefficients and intercept
        """
        return [model.coef_, model.intercept_]
    
    @staticmethod
    def set_parameters(model: SGDRegressor, parameters: List[np.ndarray]) -> None:
        """
        Set parameters in a scikit-learn model.
        
        Args:
            model: The model to set parameters in
            parameters: List containing coefficients and intercept
        """
        model.coef_, model.intercept_ = parameters
    
    @staticmethod
    def aggregate_parameters(params1: List[np.ndarray], params2: List[np.ndarray]) -> List[np.ndarray]:
        """
        Aggregate parameters from two clients using simple averaging.
        
        Args:
            params1: Parameters from first client
            params2: Parameters from second client
            
        Returns:
            Aggregated parameters
        """
        aggregated_coef = (params1[0] + params2[0]) / 2
        aggregated_intercept = (params1[1] + params2[1]) / 2
        return [aggregated_coef, aggregated_intercept] 