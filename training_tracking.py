"""
Training and Tracking Module

This module contains classes responsible for tracking training progress,
managing loss values, and executing individual training rounds in
federated learning scenarios.
"""

from typing import List, Tuple, TYPE_CHECKING
from model_management import ParameterManager

if TYPE_CHECKING:
    from federated_client import SklearnClient


class LossTracker:
    """
    Tracks and manages loss values during training.
    
    This class maintains loss history for individual clients and
    aggregated losses. It provides methods for recording losses
    and calculating aggregated loss values across multiple clients.
    """
    
    def __init__(self):
        """Initialize the loss tracker."""
        self.client_losses = []
        self.aggregated_losses = []
    
    def record_client_loss(self, loss: float) -> None:
        """
        Record a loss value for a client.
        
        Args:
            loss: Loss value to record
        """
        self.client_losses.append(loss)
    
    def record_aggregated_loss(self, loss: float) -> None:
        """
        Record an aggregated loss value.
        
        Args:
            loss: Aggregated loss value to record
        """
        self.aggregated_losses.append(loss)
    
    def get_client_losses(self) -> List[float]:
        """
        Get all recorded client losses.
        
        Returns:
            List of client loss values
        """
        return self.client_losses
    
    def get_aggregated_losses(self) -> List[float]:
        """
        Get all recorded aggregated losses.
        
        Returns:
            List of aggregated loss values
        """
        return self.aggregated_losses


class TrainingRound:
    """
    Represents a single training round in federated learning.
    
    This class encapsulates the logic for a single round of federated
    learning, including parameter collection, aggregation, distribution,
    and evaluation. It provides a clean interface for round execution.
    """
    
    def __init__(self, round_number: int):
        """
        Initialize a training round.
        
        Args:
            round_number: The round number for this training round
        """
        self.round_number = round_number
    
    def execute(self, client1: 'SklearnClient', client2: 'SklearnClient') -> Tuple[float, float, float]:
        """
        Execute a single training round.
        
        Args:
            client1: First federated learning client
            client2: Second federated learning client
            
        Returns:
            Tuple of (client1_loss, client2_loss, aggregated_loss)
        """
        print(f"\n--- Round {self.round_number} ---")
        
        params1 = client1.get_parameters({})
        params2 = client2.get_parameters({})
        
        aggregated_params = ParameterManager.aggregate_parameters(params1, params2)
        
        client1.fit(aggregated_params, {})
        client2.fit(aggregated_params, {})
        
        loss1, _, _ = client1.evaluate(aggregated_params, {})
        loss2, _, _ = client2.evaluate(aggregated_params, {})
        
        aggregated_loss = (loss1 + loss2) / 2
        
        print(f"Round {self.round_number} - Client 1: {loss1:.4f}, Client 2: {loss2:.4f}, Aggregated: {aggregated_loss:.4f}")
        
        return loss1, loss2, aggregated_loss 