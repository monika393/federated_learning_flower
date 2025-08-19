"""
Data Management Module

This module contains classes responsible for loading and preprocessing datasets
for federated learning scenarios. It provides clean interfaces for data access
and partitioning.
"""

import numpy as np
from typing import Tuple
from sklearn.datasets import load_diabetes


class DatasetLoader:
    """
    Responsible for loading and managing datasets.
    
    This class handles the loading of datasets from various sources
    and provides a clean interface for data access. It encapsulates
    all dataset-specific logic and provides methods for data retrieval.
    """
    
    @staticmethod
    def load_diabetes_dataset() -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the diabetes dataset from scikit-learn.
        
        Returns:
            Tuple of (features, targets)
        """
        print("Loading diabetes dataset...")
        return load_diabetes(return_X_y=True)


class DataSplitter:
    """
    Handles dataset partitioning for federated learning scenarios.
    
    This class is responsible for splitting datasets into appropriate
    portions for different clients in federated learning. It ensures
    that data is distributed fairly and maintains data integrity
    during the splitting process.
    """
    
    @staticmethod
    def split_for_two_clients(X: np.ndarray, y: np.ndarray, 
                             split_point: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split dataset into two parts for federated learning clients.
        
        Args:
            X: Feature matrix
            y: Target values
            split_point: Index where to split the dataset
            
        Returns:
            Tuple of (X1, y1, X2, y2) for two clients
        """
        X1, y1 = X[:split_point], y[:split_point]
        X2, y2 = X[split_point:], y[split_point:]
        print(f"Dataset split: Client 1 has {len(X1)} samples, Client 2 has {len(X2)} samples")
        return X1, y1, X2, y2 