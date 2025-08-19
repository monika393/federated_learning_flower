# Flower Federated learning ML Demo

This repository contains a modular Python implementation that demonstrates federated learning using a simplified approach that doesn't require complex Flower framework setup. The implementation shows how federated learning works by simulating the training process manually using pandas and scikit-learn.

## What Is Federated Learning?

Federated Learning allows training machine learning models without centralizing data. Instead of gathering data on one server, the computation is moved to where the data resides. In classic centralized learning, data is collected from devices and sent to a central server. In contrast, federated learning reverses this approach—the model is sent to the data.

### Why It Matters

- **Privacy & regulation**: Laws like GDPR or CCPA restrict data sharing.
- **User expectations**: Users expect sensitive information (e.g. typing patterns) not to leave their device.
- **Large-scale, distributed data**: Devices or organizations may produce too much data for centralized processing.

### Federated Learning in 5 Steps:

1. **Initialize** a global model on the server.
2. **Send** the model to selected client nodes (devices or organizations).
3. **Train** locally on each client's data for a short time (e.g. one epoch or a few mini-batches).
4. **Return** updates (model weights or gradients) to the server.
5. **Aggregate** updates via strategies like FedAvg, typically weighted by client data size.
6. **Repeat** until convergence.

## Architecture Diagrams

This repository includes comprehensive architecture diagrams that illustrate different aspects of federated learning systems:

### Basic Architecture
![Basic Architecture](flower-architecture/flower-architecture-basic-architecture.svg)

The basic architecture diagram shows the fundamental components of a federated learning system:
- **Server**: Central coordinator that manages the federated learning process
- **Clients**: Distributed nodes that perform local training
- **Communication**: Secure parameter exchange between server and clients

### Hub-and-Spoke Model
![Hub and Spoke](flower-architecture/flower-architecture-hub-and-spoke.svg)

The hub-and-spoke architecture demonstrates:
- **Central Hub**: Single server that coordinates all clients
- **Spoke Clients**: Multiple clients connected to the central hub
- **Scalability**: Easy to add or remove clients without affecting others

### Multi-Run Architecture
![Multi-Run](flower-architecture/flower-architecture-multi-run.svg)

The multi-run architecture shows:
- **Concurrent Projects**: Multiple federated learning projects running simultaneously
- **Resource Sharing**: Efficient use of infrastructure across different workloads
- **Isolation**: Each project maintains its own data and model privacy

## Overview

The demo implements a federated learning scenario where:
- Two clients train local models on separate portions of the diabetes dataset
- A central server aggregates the model parameters using FedAvg strategy
- The process preserves data privacy by keeping data local to each client
- Results are compared with a centralized training approach
- Comprehensive training statistics and convergence analysis are provided

## Architecture

The code is organized into modular files with focused responsibilities:

### Core Modules

- **`data_management.py`** - Dataset loading and partitioning
  - `DatasetLoader`: Handles dataset loading from various sources
  - `DataSplitter`: Manages dataset partitioning for federated learning

- **`model_management.py`** - Model creation and parameter management
  - `ModelFactory`: Creates and configures different ML models
  - `ParameterManager`: Handles parameter extraction, setting, and aggregation

- **`training_tracking.py`** - Training progress tracking
  - `LossTracker`: Manages loss history for clients and aggregated losses
  - `TrainingRound`: Encapsulates logic for single federated learning rounds

- **`federated_client.py`** - Core federated learning components
  - `SklearnClient`: Flower client implementation with SGDRegressor
  - `FedAvgWithLossTracking`: Enhanced federated averaging strategy

- **`model_evaluation.py`** - Model evaluation and comparison
  - `CentralizedTrainer`: Handles centralized model training for comparison
  - `FederatedModelEvaluator`: Evaluates federated models on full datasets

- **`visualization.py`** - Plotting and file management
  - `PlotCreator`: Creates various types of plots for visualization
  - `FileSaver`: Manages saving plots and results to files

- **`flower_privacy_ml_demo_simple.py`** - Main execution script with comprehensive training statistics

### Key Features

- **Modular Design**: Clean separation of concerns with dedicated files for each component
- **Type Hints**: Full type annotations for better code documentation and IDE support
- **Comprehensive Documentation**: Detailed docstrings for all classes and methods
- **Privacy Preservation**: Data remains on local clients, only model parameters are shared
- **Federated Averaging**: Uses FedAvg strategy for parameter aggregation
- **Advanced Analytics**: Detailed training statistics and convergence analysis
- **Performance Comparison**: Compares federated learning with centralized training
- **Automated Results**: Saves all graphs and metrics automatically

## Files

- `flower_privacy_ml_demo_simple.py` - Main execution script with comprehensive training statistics
- `data_management.py` - Dataset loading and partitioning
- `model_management.py` - Model creation and parameter management
- `training_tracking.py` - Training progress tracking
- `federated_client.py` - Core federated learning components
- `model_evaluation.py` - Model evaluation and comparison
- `visualization.py` - Plotting and file management
- `config.py` - Configuration settings for federated learning parameters
- `Flower-Privacy-Preserving-ML.ipynb` - Original Jupyter notebook
- `requirements.txt` - Python dependencies
- `flower-architecture/` - Directory containing architecture diagrams
- `results/` - Directory where graphs and results are saved

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main Python script:
```bash
python flower_federated_ml_demo_simple.py
```

## Output

The script will:
1. Load and split the diabetes dataset between two clients
2. Run federated learning for 5 rounds with detailed tracking
3. Generate and save the following files in the `results/` folder:
   - `training_statistics.png` - Comprehensive training metrics over rounds
   - `convergence_analysis.png` - Convergence behavior analysis
   - `model_comparison.png` - Performance comparison between centralized and federated models
   - `federated_results.txt` - Detailed training results and metrics
   - `training_statistics.csv` - Raw training data for further analysis

## Code Structure

### Data Management
```python
from data_management import DatasetLoader, DataSplitter
dataset_loader = DatasetLoader()
data_splitter = DataSplitter()
X, y = dataset_loader.load_diabetes_dataset()
X1, y1, X2, y2 = data_splitter.split_for_two_clients(X, y)
```

### Federated Learning
```python
from flower_privacy_ml_demo_simple import SimpleFederatedClient, SimpleFederatedServer
client1 = SimpleFederatedClient(X1, y1, client_id=1)
client2 = SimpleFederatedClient(X2, y2, client_id=2)
server = SimpleFederatedServer([client1, client2])
federated_history = server.run_federated_learning(num_rounds=5)
```

### Model Evaluation
```python
from model_evaluation import CentralizedTrainer, FederatedModelEvaluator
centralized_trainer = CentralizedTrainer()
federated_evaluator = FederatedModelEvaluator()
_, centralized_mse = centralized_trainer.train_gradient_boosting(X, y)
federated_mse = federated_evaluator.evaluate_aggregated_model(client1, client2, X, y)
```

### Visualization
```python
from flower_privacy_ml_demo_simple import create_training_statistics_plots, create_convergence_analysis
create_training_statistics_plots([client1, client2])
create_convergence_analysis([client1, client2])
```

## Technical Details

- **Dataset**: Diabetes dataset from scikit-learn
- **Models**: SGDRegressor for federated learning, GradientBoostingRegressor for centralized comparison
- **Framework**: Simplified federated learning implementation using pandas and scikit-learn
- **Visualization**: Matplotlib and Seaborn for creating comprehensive graphs
- **Type Safety**: Full type hints throughout the codebase
- **Analysis**: Advanced training statistics including MSE, R², MAE, and model complexity tracking

## Results Interpretation

The generated plots show:
1. **Training Statistics**: MSE, R², MAE progression over training rounds
2. **Model Complexity**: Coefficient norm and intercept evolution
3. **Convergence Analysis**: Log-scale MSE progression and improvement rates
4. **Performance Comparison**: Final round comparison between clients
5. **Performance Distribution**: Box plots showing final rounds performance

This demonstrates how federated learning can achieve competitive performance while maintaining data privacy.

## Benefits of Modular File Structure

- **Maintainability**: Each file has a focused responsibility
- **Testability**: Individual modules can be tested in isolation
- **Reusability**: Modules can be easily reused in other projects
- **Readability**: Clear file organization makes the code easier to understand
- **Extensibility**: New features can be added by creating new modules
- **Collaboration**: Multiple developers can work on different modules simultaneously
- **Debugging**: Issues can be isolated to specific modules

## Privacy Benefits

The implementation demonstrates key privacy advantages:
- **Data Localization**: Client data never leaves the client's system
- **Parameter Sharing**: Only model parameters are shared, not raw data
- **No Data Centralization**: No single point of data collection
- **Privacy-Preserving Training**: Models can be trained without compromising data privacy 
