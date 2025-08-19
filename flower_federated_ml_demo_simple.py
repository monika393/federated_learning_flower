#!/usr/bin/env python3
"""
Flower Privacy-Preserving ML Demo - Simplified Version

This module demonstrates federated learning concepts using pandas and scikit-learn
without requiring complex Flower setup. It shows how federated learning works
by simulating the training process manually.

The implementation includes:
- SimpleFederatedClient: Simulates a federated learning client with local training
- SimpleFederatedServer: Manages federated learning rounds and model aggregation
- Training statistics visualization and analysis
- Convergence analysis and performance comparison
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os
from typing import Tuple, List, Dict, Any

from data_management import DatasetLoader, DataSplitter
from model_evaluation import CentralizedTrainer, FederatedModelEvaluator
from visualization import PlotCreator, FileSaver, create_training_statistics_plots, create_convergence_analysis
from config import federated_config


class SimpleFederatedClient:
    """
    A simple federated learning client that simulates local training.
    
    This class represents a federated learning client that maintains local data
    and a local model. It can train the model on local data and participate
    in federated learning rounds by sharing model parameters.
    
    Attributes:
        X: Training features for the client
        y: Training labels for the client
        client_id: Unique identifier for the client
        model: Local SGD regressor model
        training_history: Historical training records
        round_metrics: Detailed metrics for each training round
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, client_id: int):
        """
        Initialize the federated learning client.
        
        Args:
            X: Training features as numpy array
            y: Training labels as numpy array
            client_id: Unique identifier for the client
        """
        self.X = X
        self.y = y
        self.client_id = client_id
        self.model = SGDRegressor(
            max_iter=federated_config.max_iter,
            tol=federated_config.tol,
            random_state=federated_config.random_state
        )
        self.training_history = []
        self.round_metrics = []
    
    def train_local(self, round_num: int = None) -> Dict[str, Any]:
        """
        Train the local model on client data and compute performance metrics.
        
        This method fits the local SGD regressor model to the client's data
        and calculates various performance metrics including MSE, R² score,
        and mean absolute error. It also stores metrics for visualization.
        
        Args:
            round_num: Current federated learning round number for tracking
            
        Returns:
            Dictionary containing training results with metrics and model parameters
        """
        self.model.fit(self.X, self.y)
        
        y_pred = self.model.predict(self.X)
        mse = mean_squared_error(self.y, y_pred)
        r2_score = 1 - (mse / np.var(self.y))
        mae = np.mean(np.abs(self.y - y_pred))
        
        self.training_history.append({
            'client_id': self.client_id,
            'mse': mse,
            'num_samples': len(self.X)
        })
        
        round_metric = {
            'round': round_num,
            'client_id': self.client_id,
            'mse': mse,
            'r2_score': r2_score,
            'mae': mae,
            'num_samples': len(self.X),
            'coefficient_norm': np.linalg.norm(self.model.coef_),
            'intercept': self.model.intercept_
        }
        self.round_metrics.append(round_metric)
        
        return {
            'client_id': self.client_id,
            'mse': mse,
            'r2_score': r2_score,
            'mae': mae,
            'num_samples': len(self.X),
            'coefficients': self.model.coef_,
            'intercept': self.model.intercept_,
            'coefficient_norm': np.linalg.norm(self.model.coef_)
        }
    
    def get_model_params(self) -> Tuple[np.ndarray, float]:
        """
        Retrieve the current model parameters.
        
        Returns:
            Tuple containing model coefficients and intercept
        """
        return self.model.coef_, self.model.intercept_
    
    def set_model_params(self, coefficients: np.ndarray, intercept: float):
        """
        Update the model parameters with new values.
        
        This method is used during federated learning to update the local model
        with aggregated parameters from the server.
        
        Args:
            coefficients: New model coefficients as numpy array
            intercept: New model intercept as float
        """
        self.model.coef_ = coefficients
        self.model.intercept_ = intercept


class SimpleFederatedServer:
    """
    A simple federated learning server that coordinates training rounds.
    
    This class manages the federated learning process by coordinating multiple
    clients, aggregating their model updates, and distributing the aggregated
    model back to all clients for the next round.
    
    Attributes:
        clients: List of federated learning clients
        round_history: Historical data from all training rounds
    """
    
    def __init__(self, clients: List[SimpleFederatedClient]):
        """
        Initialize the federated learning server.
        
        Args:
            clients: List of federated learning client instances
        """
        self.clients = clients
        self.round_history = []
    
    def aggregate_models(self, client_results: List[Dict[str, Any]]) -> Tuple[np.ndarray, float]:
        """
        Aggregate model parameters from multiple clients using weighted averaging.
        
        This method implements FedAvg (Federated Averaging) algorithm by computing
        a weighted average of model parameters based on the number of samples
        each client used for training.
        
        Args:
            client_results: List of dictionaries containing client training results
            
        Returns:
            Tuple containing aggregated coefficients and intercept
        """
        total_samples = sum(result['num_samples'] for result in client_results)
        
        aggregated_coefficients = np.zeros_like(client_results[0]['coefficients'])
        aggregated_intercept = 0.0
        
        for result in client_results:
            weight = result['num_samples'] / total_samples
            aggregated_coefficients += weight * result['coefficients']
            aggregated_intercept += weight * result['intercept']
        
        return aggregated_coefficients, aggregated_intercept
    
    def run_federated_learning(self, num_rounds: int) -> List[Dict[str, Any]]:
        """
        Execute federated learning for the specified number of rounds.
        
        This method orchestrates the complete federated learning process:
        1. Each client trains locally on their data
        2. Model parameters are aggregated using weighted averaging
        3. Aggregated model is distributed back to all clients
        4. Process repeats for the specified number of rounds
        
        Args:
            num_rounds: Number of federated learning rounds to execute
            
        Returns:
            List of dictionaries containing round-by-round results and metrics
        """
        print(f"Starting federated learning with {len(self.clients)} clients for {num_rounds} rounds...")
        
        for round_num in range(num_rounds):
            print(f"   Round {round_num + 1}/{num_rounds}")
            
            client_results = []
            for client in self.clients:
                result = client.train_local(round_num + 1)
                client_results.append(result)
                print(f"     Client {client.client_id}: MSE = {result['mse']:.4f}, R² = {result['r2_score']:.4f}")
            
            aggregated_coefficients, aggregated_intercept = self.aggregate_models(client_results)
            
            for client in self.clients:
                client.set_model_params(aggregated_coefficients, aggregated_intercept)
            
            round_metrics = {
                'round': round_num + 1,
                'client_results': client_results,
                'aggregated_coefficients': aggregated_coefficients,
                'aggregated_intercept': aggregated_intercept
            }
            self.round_history.append(round_metrics)
            
            avg_mse = np.mean([result['mse'] for result in client_results])
            avg_r2 = np.mean([result['r2_score'] for result in client_results])
            print(f"     Average MSE: {avg_mse:.4f}, Average R²: {avg_r2:.4f}")
        
        print("Federated learning completed!")
        return self.round_history


def main():
    """
    Main execution function for the simplified federated learning demo.
    
    This function orchestrates the complete federated learning experiment:
    1. Loads and splits the diabetes dataset
    2. Creates federated learning clients and server
    3. Executes federated learning rounds
    4. Compares with centralized training
    5. Generates visualizations and saves results
    """
    print("Starting Simplified Flower Privacy-Preserving ML Demo")
    print("=" * 60)
    
    plot_creator = PlotCreator()
    file_saver = FileSaver(federated_config.results_dir)
    
    print("Loading and splitting dataset...")
    dataset_loader = DatasetLoader()
    data_splitter = DataSplitter()
    
    X, y = dataset_loader.load_diabetes_dataset()
    X1, y1, X2, y2 = data_splitter.split_for_two_clients(X, y, federated_config.split_point)
    
    print(f"Dataset loaded: {len(X)} samples")
    print(f"   Client 1: {len(X1)} samples")
    print(f"   Client 2: {len(X2)} samples")
    print("-" * 50)
    
    print("Creating federated learning clients...")
    client1 = SimpleFederatedClient(X1, y1, client_id=1)
    client2 = SimpleFederatedClient(X2, y2, client_id=2)
    
    server = SimpleFederatedServer([client1, client2])
    
    federated_history = server.run_federated_learning(federated_config.num_rounds)
    
    print("\nFederated learning completed!")
    print("=" * 50)
    
    print("\nTraining centralized model for comparison...")
    centralized_trainer = CentralizedTrainer()
    _, centralized_mse = centralized_trainer.train_gradient_boosting(X, y)
    
    print("\nEvaluating federated model...")
    federated_model = SGDRegressor()
    federated_model.coef_ = client1.get_model_params()[0]
    federated_model.intercept_ = client1.get_model_params()[1]
    
    y_pred_federated = federated_model.predict(X)
    federated_mse = mean_squared_error(y, y_pred_federated)
    
    print(f"   Centralized GBR MSE: {centralized_mse:.4f}")
    print(f"   Federated SGD MSE: {federated_mse:.4f}")
    print(f"   Performance difference: {abs(centralized_mse - federated_mse):.4f}")
    
    if federated_config.save_plots or federated_config.show_plots:
        print("\nCreating visualizations...")
        
        print("   Creating training statistics plots...")
        if federated_config.show_plots:
            create_training_statistics_plots([client1, client2])
        
        if federated_config.save_plots:
            stats_plot_path = os.path.join(federated_config.results_dir, 'training_statistics.png')
            create_training_statistics_plots([client1, client2], stats_plot_path)
        
        print("   Creating convergence analysis plots...")
        if federated_config.show_plots:
            create_convergence_analysis([client1, client2])
        
        if federated_config.save_plots:
            convergence_plot_path = os.path.join(federated_config.results_dir, 'convergence_analysis.png')
            create_convergence_analysis([client1, client2], convergence_plot_path)
        
        if federated_config.show_plots:
            plot_creator.create_model_comparison_plot(centralized_mse, federated_mse)
        
        if federated_config.save_plots:
            file_saver.save_comparison_plot(centralized_mse, federated_mse)
    
    print("\nSaving results...")
    
    results = {
        'centralized_mse': centralized_mse,
        'federated_mse': federated_mse,
        'federated_rounds': len(federated_history),
        'client1_samples': len(X1),
        'client2_samples': len(X2),
        'total_samples': len(X)
    }
    
    results_file = os.path.join(federated_config.results_dir, 'federated_results.txt')
    os.makedirs(federated_config.results_dir, exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write("Federated Learning Results\n")
        f.write("=" * 30 + "\n")
        f.write(f"Centralized GBR MSE: {centralized_mse:.4f}\n")
        f.write(f"Federated SGD MSE: {federated_mse:.4f}\n")
        f.write(f"Performance difference: {abs(centralized_mse - federated_mse):.4f}\n")
        f.write(f"Federated rounds: {len(federated_history)}\n")
        f.write(f"Client 1 samples: {len(X1)}\n")
        f.write(f"Client 2 samples: {len(X2)}\n")
        f.write(f"Total samples: {len(X)}\n\n")
        
        f.write("Detailed Training Statistics\n")
        f.write("-" * 25 + "\n")
        
        for client in [client1, client2]:
            f.write(f"\nClient {client.client_id}:\n")
            f.write(f"  Total samples: {len(client.X)}\n")
            f.write(f"  Final MSE: {client.round_metrics[-1]['mse']:.4f}\n")
            f.write(f"  Final R²: {client.round_metrics[-1]['r2_score']:.4f}\n")
            f.write(f"  Final MAE: {client.round_metrics[-1]['mae']:.4f}\n")
            f.write(f"  Final coefficient norm: {client.round_metrics[-1]['coefficient_norm']:.4f}\n")
            
            if len(client.round_metrics) > 1:
                first_mse = client.round_metrics[0]['mse']
                last_mse = client.round_metrics[-1]['mse']
                improvement = ((first_mse - last_mse) / first_mse) * 100
                f.write(f"  MSE improvement: {improvement:.2f}%\n")
    
    import pandas as pd
    all_training_data = []
    for client in [client1, client2]:
        all_training_data.extend(client.round_metrics)
    
    training_df = pd.DataFrame(all_training_data)
    training_csv_path = os.path.join(federated_config.results_dir, 'training_statistics.csv')
    training_df.to_csv(training_csv_path, index=False)
    print(f"Detailed training statistics saved to: {training_csv_path}")
    
    print("\nAll results have been saved to the 'results' folder!")
    print("\nSummary:")
    print(f"   Centralized GBR MSE: {centralized_mse:.4f}")
    print(f"   Federated SGD MSE: {federated_mse:.4f}")
    print(f"   Performance difference: {abs(centralized_mse - federated_mse):.4f}")
    
    if federated_mse < centralized_mse:
        print(f"   Federated learning performed better!")
    elif centralized_mse < federated_mse:
        print(f"   Centralized learning performed better")
    else:
        print(f"   Both approaches performed similarly")
    
    print(f"\nPrivacy Benefits:")
    print(f"   - Client 1 data never left client 1's system")
    print(f"   - Client 2 data never left client 2's system")
    print(f"   - Only model parameters were shared, not raw data")
    print(f"   - Federated learning achieved {federated_mse:.4f} MSE without data sharing")


if __name__ == "__main__":
    main() 