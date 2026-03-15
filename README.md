# Satellite Telemetry

An LSTM-based autoencoder for detecting anomalies in satellite telemetry data with uncertainty quantification through Monte Carlo Dropout.

## 📋 Overview

This project implements an LSTM autoencoder model designed to detect anomalies in satellite telemetry channels (specifically P-1 channel data). The model uses Monte Carlo Dropout for uncertainty estimation and is evaluated against labeled anomalies to assess detection performance.

## 🎯 Features

- **LSTM Autoencoder Architecture**: Deep learning model for unsupervised anomaly detection
- **Monte Carlo Dropout**: Uncertainty quantification for robust anomaly detection
- **Multi-Seed Experiments**: Reproducible results across multiple random seeds
- **Comprehensive Evaluation**: Metrics including ROC-AUC, precision, recall, and F1-scores
- **Results Aggregation**: Automated aggregation and visualization of multi-seed results
- **PyTorch Implementation**: Efficient GPU-enabled training and inference

## 📁 Project Structure

```
satellite-telemetry/
├── main.py                          # Main entry point with CLI
├── requirements.txt                 # Python dependencies (pip)
├── data/                            # Data directory
│   └── labeled_anomalies.csv       # Labeled anomaly data
├── results/                         # Output directory for results
│   ├── aggregated/                  # Aggregated results across all seeds
│   ├── seed_0/                      # Results for seed 0
│   ├── seed_1/                      # Results for seed 1
│   ├── seed_2/                      # Results for seed 2
│   ├── seed_3/                      # Results for seed 3
│   ├── seed_4/                      # Results for seed 4
│   └── seed_42/                     # Results for seed 42
├── src/                             # Source code directory
│   ├── __init__.py                  # Package initialization
│   ├── utils.py                     # Utility functions (seed setting, etc.)
│   ├── data_loader.py               # Data loading and preprocessing
│   ├── model.py                     # LSTM Autoencoder model definition
│   ├── train.py                     # Training pipeline
│   ├── evaluate.py                  # Evaluation metrics and procedures
│   ├── aggregate_results.py         # Multi-seed results aggregation
│   ├── mc_dropout.py                # Monte Carlo Dropout utilities
│   ├── metrics.py                   # Custom evaluation metrics
│   ├── visualization.py             # Plotting and visualization utilities
│   ├── simulation.py                # Simulation utilities
│   ├── statistics.py                # Statistical analysis tools
│   ├── generate_results_table.py    # Results table generation
│   └── run_significance.py          # Significance testing
└── .gitignore

### Requirements

The project dependencies are specified in `requirements.txt`:

```
# Core numerical & data handling
numpy>=1.23
pandas>=1.5

# Machine learning & evaluation
scikit-learn>=1.2

# Deep learning (LSTM Autoencoder + MC Dropout)
torch>=2.0
torchvision>=0.15

# Visualization
matplotlib>=3.7
seaborn>=0.12

# Utilities
tqdm>=4.65
pyyaml>=6.0
```

### Results Directory Structure

The `results/` directory contains outputs from multi-seed experiments:

- **aggregated/**: Contains aggregated statistics and analyses across all seed experiments
- **seed_0/, seed_1/, seed_2/, seed_3/, seed_4/, seed_42/**: Individual directories for each seed's training and evaluation results, including models, metrics, and visualizations

```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Keyan1122/satellite-telemetry.git
cd satellite-telemetry

# Install dependencies
pip install -r requirements.txt
```

### Usage

The project provides a CLI interface with multiple modes:

#### Train a single model
```bash
python main.py --mode train --seed 0 --device cuda
```

#### Evaluate an existing model
```bash
python main.py --mode evaluate --seed 0 --device cuda
```

#### Run complete multi-seed experiment
```bash
python main.py --mode full_experiment --seeds 5 --device cuda
```

#### Aggregate results from multiple seeds
```bash
python main.py --mode aggregate
```

#### Generate visualization plots
```bash
python main.py --mode plot --seed 0
```

## 🔧 Configuration

Key hyperparameters in `main.py`:

- `window_size`: 50 (number of timesteps per sample)
- `stride`: 1 (sliding window stride)
- `hidden_dim`: 64 (LSTM hidden dimension)
- `latent_dim`: 16 (autoencoder latent space dimension)
- `dropout`: 0.2 (MC Dropout rate)
- `channel_id`: "P-1" (satellite channel to analyze)

## 📊 Model Architecture

**LSTM Autoencoder**:
- **Encoder**: LSTM layers that compress input sequences into a latent representation
- **Decoder**: LSTM layers that reconstruct the input from the latent code
- **MC Dropout**: Applied after encoder for uncertainty estimation

### Evaluation Modes

1. **MC Dropout Mode**: Uses dropout during inference to estimate uncertainty
2. **Standard Mode**: Evaluation without dropout (deterministic)

## 📈 Outputs

Results are saved in the `results/` directory:
- Training logs and metrics per seed
- Evaluation metrics (ROC-AUC, precision, recall, F1)
- Aggregated statistics across seeds
- Visualization plots

## 🔬 Methodology

1. **Data Preparation**: Load satellite telemetry and create sliding windows
2. **Model Training**: Train LSTM autoencoder on normal (non-anomalous) data
3. **Anomaly Detection**: Compute reconstruction error; high error indicates anomalies
4. **Uncertainty Estimation**: MC Dropout provides confidence intervals around predictions
5. **Evaluation**: Compare predictions against labeled anomalies

## 📝 Data Format

Expected data format:
- Satellite telemetry CSV files with time series values
- Labeled anomalies CSV with timestamps and anomaly labels
- Channel-specific data (default: P-1)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is open source and available under the MIT License.

## ✉️ Contact

For questions or inquiries, please reach out to the repository maintainer.