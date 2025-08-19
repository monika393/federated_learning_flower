"""
Visualization Module

This module contains classes and functions responsible for creating plots and saving
results to files for the federated learning experiment. It includes comprehensive
training statistics visualization and convergence analysis capabilities.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from federated_client import SklearnClient


class PlotCreator:
    """
    Creates various types of plots for visualization.
    
    This class handles the creation of different types of plots
    including loss progression plots and model comparison charts.
    It provides a clean interface for generating visualizations
    with consistent styling and formatting.
    """
    
    @staticmethod
    def create_loss_progression_plot(client1: 'SklearnClient', client2: 'SklearnClient', 
                                   global_losses: List[float]) -> None:
        """
        Create a plot showing loss progression across training rounds.
        
        Args:
            client1: First client with loss history
            client2: Second client with loss history
            global_losses: List of aggregated losses
        """
        print("Creating loss plot...")
        rounds = range(1, len(global_losses) + 1)

        plt.figure(figsize=(12, 8))
        plt.plot(rounds, client1.loss_tracker.get_client_losses(), label="Client 1 Loss", marker='o', linewidth=2)
        plt.plot(rounds, client2.loss_tracker.get_client_losses(), label="Client 2 Loss", marker='s', linewidth=2)
        plt.plot(rounds, global_losses, label="Server Aggregated Loss", marker='^', linewidth=3)

        plt.title("Federated Learning: Client & Server Loss per Round", fontsize=16, fontweight='bold')
        plt.xlabel("Round", fontsize=14)
        plt.ylabel("Mean Squared Error (MSE)", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def create_model_comparison_plot(centralized_mse: float, federated_mse: float) -> None:
        """
        Create a bar chart comparing model performances.
        
        Args:
            centralized_mse: MSE score of centralized model
            federated_mse: MSE score of federated model
        """
        print("Creating model comparison plot...")
        models = ['Centralized GBR', 'Federated SGD']
        mse_values = [centralized_mse, federated_mse]
        colors = ['#2E86AB', '#A23B72']

        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, mse_values, color=colors, alpha=0.8)
        plt.title("Model Performance Comparison", fontsize=16, fontweight='bold')
        plt.ylabel("Mean Squared Error (MSE)", fontsize=14)
        plt.ylim(0, max(mse_values) * 1.1)

        for bar, value in zip(bars, mse_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()


class FileSaver:
    """
    Handles saving of plots and results to files.
    
    This class manages the file system operations for saving
    generated plots and result files. It ensures proper directory
    structure and file naming conventions are maintained.
    """
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize the file saver.
        
        Args:
            results_dir: Directory to save outputs
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def save_loss_plot(self, client1: 'SklearnClient', client2: 'SklearnClient', 
                      global_losses: List[float]) -> str:
        """
        Save the loss progression plot to file.
        
        Args:
            client1: First client with loss history
            client2: Second client with loss history
            global_losses: List of aggregated losses
            
        Returns:
            Path to the saved plot file
        """
        rounds = range(1, len(global_losses) + 1)

        plt.figure(figsize=(12, 8))
        plt.plot(rounds, client1.loss_tracker.get_client_losses(), label="Client 1 Loss", marker='o', linewidth=2)
        plt.plot(rounds, client2.loss_tracker.get_client_losses(), label="Client 2 Loss", marker='s', linewidth=2)
        plt.plot(rounds, global_losses, label="Server Aggregated Loss", marker='^', linewidth=3)

        plt.title("Federated Learning: Client & Server Loss per Round", fontsize=16, fontweight='bold')
        plt.xlabel("Round", fontsize=14)
        plt.ylabel("Mean Squared Error (MSE)", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(self.results_dir, "federated_learning_loss.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Loss plot saved to: {plot_path}")
        plt.close()
        return plot_path

    def save_comparison_plot(self, centralized_mse: float, federated_mse: float) -> str:
        """
        Save the model comparison plot to file.
        
        Args:
            centralized_mse: MSE score of centralized model
            federated_mse: MSE score of federated model
            
        Returns:
            Path to the saved plot file
        """
        models = ['Centralized GBR', 'Federated SGD']
        mse_values = [centralized_mse, federated_mse]
        colors = ['#2E86AB', '#A23B72']

        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, mse_values, color=colors, alpha=0.8)
        plt.title("Model Performance Comparison", fontsize=16, fontweight='bold')
        plt.ylabel("Mean Squared Error (MSE)", fontsize=14)
        plt.ylim(0, max(mse_values) * 1.1)

        for bar, value in zip(bars, mse_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        comparison_plot_path = os.path.join(self.results_dir, "model_comparison.png")
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to: {comparison_plot_path}")
        plt.close()
        return comparison_plot_path

    def save_training_results(self, client1: 'SklearnClient', client2: 'SklearnClient', 
                            X, global_losses: List[float], 
                            centralized_mse: float, federated_mse: float) -> str:
        """
        Save detailed training results to a text file.
        
        Args:
            client1: First client with training history
            client2: Second client with training history
            X: Full dataset
            global_losses: List of aggregated losses
            centralized_mse: MSE score of centralized model
            federated_mse: MSE score of federated model
            
        Returns:
            Path to the saved results file
        """
        results_path = os.path.join(self.results_dir, "training_results.txt")
        with open(results_path, 'w') as f:
            f.write("Flower Privacy-Preserving ML Demo Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dataset: Diabetes dataset\n")
            f.write(f"Client 1 samples: {len(client1.X)}\n")
            f.write(f"Client 2 samples: {len(client2.X)}\n")
            f.write(f"Total samples: {len(X)}\n\n")
            
            f.write("Training Results:\n")
            f.write("-" * 20 + "\n")
            for i, (c1_loss, c2_loss, global_loss) in enumerate(zip(client1.loss_tracker.get_client_losses(), 
                                                                   client2.loss_tracker.get_client_losses(), 
                                                                   global_losses)):
                f.write(f"Round {i+1}: Client1={c1_loss:.4f}, Client2={c2_loss:.4f}, Global={global_loss:.4f}\n")
            
            f.write(f"\nFinal Model Performance:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Centralized Gradient Boosting Regressor MSE: {centralized_mse:.4f}\n")
            f.write(f"Federated SGDRegressor MSE: {federated_mse:.4f}\n")
        
        print(f"Training results saved to: {results_path}")
        return results_path


def create_training_statistics_plots(clients: List, save_path: str = None):
    """
    Create comprehensive training statistics plots for federated learning.
    
    This function generates a multi-panel visualization showing various training
    metrics over federated learning rounds, including MSE progression, R² scores,
    model complexity evolution, and final performance comparison.
    
    Args:
        clients: List of federated learning client instances
        save_path: Optional file path to save the generated plot
    """
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Federated Learning Training Statistics', fontsize=16, fontweight='bold')
    
    all_metrics = []
    for client in clients:
        all_metrics.extend(client.round_metrics)
    
    if not all_metrics:
        print("No training metrics available for plotting")
        return
    
    df = pd.DataFrame(all_metrics)
    
    ax1 = axes[0, 0]
    for client_id in df['client_id'].unique():
        client_data = df[df['client_id'] == client_id]
        ax1.plot(client_data['round'], client_data['mse'], 
                marker='o', linewidth=2, label=f'Client {client_id}')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Mean Squared Error (MSE)')
    ax1.set_title('MSE Progression Over Training Rounds')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    for client_id in df['client_id'].unique():
        client_data = df[df['client_id'] == client_id]
        ax2.plot(client_data['round'], client_data['r2_score'], 
                marker='s', linewidth=2, label=f'Client {client_id}')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('R² Score')
    ax2.set_title('R² Score Progression Over Training Rounds')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[0, 2]
    for client_id in df['client_id'].unique():
        client_data = df[df['client_id'] == client_id]
        ax3.plot(client_data['round'], client_data['mae'], 
                marker='^', linewidth=2, label=f'Client {client_id}')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Mean Absolute Error (MAE)')
    ax3.set_title('MAE Progression Over Training Rounds')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 0]
    for client_id in df['client_id'].unique():
        client_data = df[df['client_id'] == client_id]
        ax4.plot(client_data['round'], client_data['coefficient_norm'], 
                marker='d', linewidth=2, label=f'Client {client_id}')
    ax4.set_xlabel('Round')
    ax4.set_ylabel('Coefficient Norm (L2)')
    ax4.set_title('Model Complexity Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    ax5 = axes[1, 1]
    for client_id in df['client_id'].unique():
        client_data = df[df['client_id'] == client_id]
        ax5.plot(client_data['round'], client_data['intercept'], 
                marker='*', linewidth=2, label=f'Client {client_id}')
    ax5.set_xlabel('Round')
    ax5.set_ylabel('Intercept')
    ax5.set_title('Model Intercept Evolution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    ax6 = axes[1, 2]
    final_round = df['round'].max()
    final_data = df[df['round'] == final_round]
    
    metrics = ['mse', 'r2_score', 'mae']
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    for i, client_id in enumerate(final_data['client_id'].unique()):
        client_final = final_data[final_data['client_id'] == client_id]
        values = [client_final[metric].iloc[0] for metric in metrics]
        ax6.bar(x_pos + i*width, values, width, label=f'Client {client_id}', alpha=0.8)
    
    ax6.set_xlabel('Metrics')
    ax6.set_ylabel('Value')
    ax6.set_title(f'Final Round ({final_round}) Performance Comparison')
    ax6.set_xticks(x_pos + width/2)
    ax6.set_xticklabels(['MSE', 'R²', 'MAE'])
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training statistics plot saved to: {save_path}")
    
    plt.show()


def create_convergence_analysis(clients: List, save_path: str = None):
    """
    Create convergence analysis plots for federated learning.
    
    This function generates specialized plots to analyze the convergence behavior
    of federated learning, including log-scale MSE progression, parameter
    convergence, performance improvement rates, and final round distributions.
    
    Args:
        clients: List of federated learning client instances
        save_path: Optional file path to save the generated plot
    """
    plt.style.use('default')
    sns.set_palette("Set2")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Federated Learning Convergence Analysis', fontsize=16, fontweight='bold')
    
    all_metrics = []
    for client in clients:
        all_metrics.extend(client.round_metrics)
    
    if not all_metrics:
        print("No training metrics available for convergence analysis")
        return
    
    df = pd.DataFrame(all_metrics)
    
    ax1 = axes[0, 0]
    for client_id in df['client_id'].unique():
        client_data = df[df['client_id'] == client_id]
        ax1.semilogy(client_data['round'], client_data['mse'], 
                    marker='o', linewidth=2, label=f'Client {client_id}')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Mean Squared Error (MSE) - Log Scale')
    ax1.set_title('Convergence Analysis: MSE Over Rounds')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    for client_id in df['client_id'].unique():
        client_data = df[df['client_id'] == client_id]
        ax2.plot(client_data['round'], client_data['coefficient_norm'], 
                marker='s', linewidth=2, label=f'Client {client_id}')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Coefficient Norm')
    ax2.set_title('Parameter Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    for client_id in df['client_id'].unique():
        client_data = df[df['client_id'] == client_id].sort_values('round')
        mse_improvement = client_data['mse'].pct_change() * 100
        ax3.plot(client_data['round'][1:], mse_improvement[1:], 
                marker='^', linewidth=2, label=f'Client {client_id}')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('MSE Improvement (%)')
    ax3.set_title('Performance Improvement Rate')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax4 = axes[1, 1]
    final_rounds = df[df['round'] >= df['round'].max() - 2]
    final_rounds.boxplot(column='mse', by='client_id', ax=ax4)
    ax4.set_xlabel('Client ID')
    ax4.set_ylabel('MSE')
    ax4.set_title('Final Rounds Performance Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence analysis plot saved to: {save_path}")
    
    plt.show() 