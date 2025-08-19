"""
Configuration Module

This module contains all configuration settings for the Flower federated learning demo.
Centralizing configuration makes the project more maintainable and configurable.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class FederatedConfig:
    """Configuration for federated learning parameters."""
    
    # Number of clients
    num_clients: int = 2
    
    # Training parameters
    num_rounds: int = 5
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    
    # Client selection parameters
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2
    
    # Model parameters
    max_iter: int = 1000
    tol: float = 1e-3
    random_state: int = 42
    
    # Dataset parameters
    split_point: int = 200
    
    # Resource configuration
    num_cpus: int = 1
    num_gpus: float = 0.0
    
    # Output configuration
    results_dir: str = "results"
    save_plots: bool = True
    show_plots: bool = True


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    
    # SGDRegressor parameters
    sgd_max_iter: int = 1000
    sgd_tol: float = 1e-3
    sgd_random_state: int = 42
    
    # GradientBoostingRegressor parameters
    gbr_random_state: int = 42
    gbr_n_estimators: int = 100
    gbr_learning_rate: float = 0.1


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    
    # Plot settings
    figsize_loss: tuple = (12, 8)
    figsize_comparison: tuple = (10, 6)
    dpi: int = 300
    
    # Colors
    colors: Dict[str, str] = None
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                'centralized': '#2E86AB',
                'federated': '#A23B72',
                'client1': '#4ECDC4',
                'client2': '#45B7D1',
                'aggregated': '#96CEB4'
            }


# Global configuration instances
federated_config = FederatedConfig()
model_config = ModelConfig()
viz_config = VisualizationConfig()


def get_backend_config() -> Dict[str, Any]:
    """
    Get backend configuration for Flower simulation.
    
    Returns:
        Backend configuration dictionary
    """
    return {
        "client_resources": {
            "num_cpus": federated_config.num_cpus,
            "num_gpus": federated_config.num_gpus
        }
    }





def get_fedavg_strategy_config() -> Dict[str, Any]:
    """
    Get FedAvg strategy configuration.
    
    Returns:
        Strategy configuration dictionary
    """
    return {
        "fraction_fit": federated_config.fraction_fit,
        "fraction_evaluate": federated_config.fraction_evaluate,
        "min_fit_clients": federated_config.min_fit_clients,
        "min_evaluate_clients": federated_config.min_evaluate_clients,
        "min_available_clients": federated_config.min_available_clients,
    } 