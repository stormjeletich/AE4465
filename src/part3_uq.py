"""
Part 3: Uncertainty Quantification for RUL Prediction
======================================================
Extends Part 2 RUL model with two Deep Learning UQ methods
based on the Nemani et al. paper:
1. MC Dropout (Monte Carlo Dropout)
2. Deep Neural Network Ensemble (Deep Ensembles)

Implemented in PyTorch for robust Windows execution.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import warnings

warnings.filterwarnings('ignore')

# Import data loading function
from import_data import load_data

# Constants
RUL_CAP = 125
FEATURE_COLUMNS_TO_EXCLUDE = ['engine', 'cycle']
OUTPUT_DIR = os.path.join("Output", "part3")

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def calculate_rul(df: pd.DataFrame) -> pd.Series:
    """Calculate RUL with 125-cycle cap for the training set."""
    max_cycles = df.groupby('engine')['cycle'].transform('max')
    rul = max_cycles - df['cycle']
    rul = np.minimum(rul, RUL_CAP)
    return rul


def preprocess_data(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """Preprocess data: remove low-variance features, standardize, load true test RUL."""
    feature_cols = [col for col in df_train.columns if col not in FEATURE_COLUMNS_TO_EXCLUDE]
    
    # Remove low-variance features
    variances = df_train[feature_cols].var()
    low_var_threshold = 0.01
    cols_to_keep = variances[variances > low_var_threshold].index.tolist()
    
    df_train_processed = df_train[cols_to_keep].copy()
    df_test_processed = df_test[cols_to_keep].copy()
    
    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train_processed)
    X_test_scaled = scaler.transform(df_test_processed)
    
    # Calculate RUL for training data
    y_train = calculate_rul(df_train).values
    
    # Extract final cycles for the test set
    test_final_indices = df_test.groupby('engine')['cycle'].idxmax()
    X_test_final = X_test_scaled[test_final_indices.values]
    
    # Load true RUL test labels
    try:
        # Check standard root-level path (if running from the main repo folder)
        rul_path = 'CMAPSSData/RUL_FD001.txt' 
        if not os.path.exists(rul_path):
            # Fallback relative path (if running directly from inside the src folder)
            rul_path = '../CMAPSSData/RUL_FD001.txt' 
            
        true_rul = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'])
        y_test = true_rul['RUL'].values
        # Cap test RUL to ensure consistent evaluation metrics
        y_test = np.minimum(y_test, RUL_CAP)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find RUL_FD001.txt! Looked in: {rul_path}")
    
    print(f"Training set shape: {X_train_scaled.shape}, Labels: {y_train.shape}")
    print(f"Test set shape: {X_test_final.shape}, Labels: {y_test.shape}")
    
    return X_train_scaled, X_test_final, y_train, y_test


# ============================================================================
# PYTORCH MODEL DEFINITIONS & TRAINING
# ============================================================================

class RULNet(nn.Module):
    """Standard Multi-Layer Perceptron. Can act as deterministic or MC Dropout."""
    def __init__(self, input_dim, use_dropout=False):
        super(RULNet, self).__init__()
        self.use_dropout = use_dropout
        
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        if self.use_dropout: x = self.dropout(x)
        
        x = self.relu(self.fc2(x))
        if self.use_dropout: x = self.dropout(x)
        
        x = self.relu(self.fc3(x))
        if self.use_dropout: x = self.dropout(x)
        
        return self.out(x)


def train_model(model, X, y, epochs=40, batch_size=256, seed=42):
    """Handles the PyTorch training loop."""
    torch.manual_seed(seed)
    
    # Convert numpy arrays to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).view(-1, 1)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train() # Set to training mode
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    return model


# ============================================================================
# UQ METHOD 1: MONTE CARLO (MC) DROPOUT
# ============================================================================

def mc_dropout_predict(model, X_test, num_passes=30):
    """Generates predictions via stochastic forward passes."""
    print(f"Running {num_passes} stochastic forward passes for MC Dropout...")
    
    # CRITICAL: We explicitly set model.train() here to force PyTorch 
    # to keep Dropout layers active during inference.
    model.train() 
    X_tensor = torch.FloatTensor(X_test)
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_passes):
            pred = model(X_tensor).numpy().flatten()
            predictions.append(pred)
            
    predictions = np.array(predictions)
    y_mean = np.mean(predictions, axis=0)
    y_var = np.var(predictions, axis=0)
    y_std = np.sqrt(y_var)
    return y_mean, y_var, y_std


# ============================================================================
# UQ METHOD 2: DEEP NEURAL NETWORK ENSEMBLE
# ============================================================================

def train_deep_ensemble(X_train, y_train, input_dim, n_models=5):
    """Trains an ensemble of MLPs, each with a different random initialization."""
    ensemble = []
    print("\n" + "-"*70)
    print(f"METHOD 2: TRAINING DEEP ENSEMBLE ({n_models} Neural Networks)")
    print("-"*70)
    
    for i in range(n_models):
        print(f"Training Network {i+1}/{n_models}...")
        model = RULNet(input_dim, use_dropout=False)
        # Multiply seed by 100 to ensure distinct initialization trajectories
        model = train_model(model, X_train, y_train, epochs=40, seed=(i+1)*100)
        ensemble.append(model)
        
    return ensemble


def deep_ensemble_predict(ensemble, X_test):
    """Generates predictions across the ensemble models."""
    print("Running inference across the ensemble...")
    X_tensor = torch.FloatTensor(X_test)
    predictions = []
    
    with torch.no_grad():
        for model in ensemble:
            model.eval() # Set to evaluation mode
            pred = model(X_tensor).numpy().flatten()
            predictions.append(pred)
            
    predictions = np.array(predictions)
    y_mean = np.mean(predictions, axis=0)
    y_var = np.var(predictions, axis=0)
    y_std = np.sqrt(y_var)
    return y_mean, y_var, y_std


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def compute_nll(y_true, y_pred_mean, y_pred_var):
    """Computes Negative Log-Likelihood assuming a Gaussian distribution."""
    y_pred_var = np.maximum(y_pred_var, 1e-6)  # Avoid division by zero
    nll = 0.5 * np.log(2 * np.pi * y_pred_var) + 0.5 * ((y_true - y_pred_mean) ** 2) / y_pred_var
    return np.mean(nll)

def compute_calibration_curve(y_true, y_pred_mean, y_pred_std, confidence_levels=None):
    """Computes expected vs. observed confidence."""
    if confidence_levels is None:
        confidence_levels = np.linspace(0.05, 0.95, 19)
    
    observed_confidence = []
    for conf in confidence_levels:
        # Two-tailed Z-score
        z_score = np.abs(np.percentile(np.random.randn(100000), 100 * (1 - conf) / 2))
        
        lower_bound = y_pred_mean - z_score * y_pred_std
        upper_bound = y_pred_mean + z_score * y_pred_std
        
        within_bounds = np.sum((y_true >= lower_bound) & (y_true <= upper_bound))
        observed_conf = within_bounds / len(y_true)
        observed_confidence.append(observed_conf)
    
    return np.array(confidence_levels), np.array(observed_confidence)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    if not os.path.exists('Output'):
        os.makedirs('Output')

    print("\n" + "="*70)
    print("PART 3: UNCERTAINTY QUANTIFICATION FOR RUL PREDICTION (PYTORCH)")
    print("="*70)
    
    # 1. Load and preprocess data
    print("\nLoading and Preprocessing C-MAPSS FD001 data...")
    df_train, df_test = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df_train, df_test)
    input_dim = X_train.shape[1]
    
    # 2. MC Dropout Training and Inference
    print("\n" + "-"*70)
    print("METHOD 1: MONTE CARLO (MC) DROPOUT")
    print("-" * 70)
    print("Training MC Dropout Model...")
    mc_model = RULNet(input_dim, use_dropout=True)
    mc_model = train_model(mc_model, X_train, y_train, epochs=40, seed=42)
    
    mc_mean, mc_var, mc_std = mc_dropout_predict(mc_model, X_test, num_passes=30)
    mc_rmse = compute_rmse(y_test, mc_mean)
    mc_nll = compute_nll(y_test, mc_mean, mc_var)
    
    # 3. Deep Ensemble Training and Inference
    de_ensemble = train_deep_ensemble(X_train, y_train, input_dim, n_models=5)
    de_mean, de_var, de_std = deep_ensemble_predict(de_ensemble, X_test)
    de_rmse = compute_rmse(y_test, de_mean)
    de_nll = compute_nll(y_test, de_mean, de_var)
    
    # 4. Calibration Curves
    print("\nComputing Calibration Curves...")
    mc_conf_levels, mc_obs_conf = compute_calibration_curve(y_test, mc_mean, mc_std)
    de_conf_levels, de_obs_conf = compute_calibration_curve(y_test, de_mean, de_std)
    
    # 5. Summary Output
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"\n{'Metric':<30} {'MC Dropout':<20} {'Deep Ensemble':<20}")
    print("-" * 70)
    print(f"{'RMSE (cycles)':<30} {mc_rmse:<20.4f} {de_rmse:<20.4f}")
    print(f"{'Negative Log-Likelihood':<30} {mc_nll:<20.4f} {de_nll:<20.4f}")
    print(f"{'Mean Uncertainty (Std Dev)':<30} {np.mean(mc_std):<20.4f} {np.mean(de_std):<20.4f}")
    
    # 6. Plotting
    print("\nGenerating comparison plots...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot A: Calibration Curves
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], 'k--', label='Ideal (Perfect Calibration)', linewidth=2)
    ax1.plot(mc_conf_levels, mc_obs_conf, 'o-', color='purple', label='MC Dropout', linewidth=2)
    ax1.plot(de_conf_levels, de_obs_conf, 's-', color='orange', label='Deep Ensemble', linewidth=2)
    ax1.set_xlabel('Expected Confidence (%)', fontsize=12)
    ax1.set_ylabel('Observed Confidence (%)', fontsize=12)
    ax1.set_title('Calibration Curve Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # Plot B: Predictions vs. Ground Truth with Uncertainty Bounds
    ax2 = axes[1]
    sorted_idx = np.argsort(y_test)
    test_indices = np.arange(len(y_test))
    
    ax2.scatter(test_indices, y_test[sorted_idx], color='black', label='Ground Truth', s=15, zorder=5)
    
    # Deep Ensemble Plotting
    ax2.plot(test_indices, de_mean[sorted_idx], '-', color='orange', label='Ensemble Mean', linewidth=1.5)
    ax2.fill_between(test_indices, 
                     de_mean[sorted_idx] - de_std[sorted_idx], 
                     de_mean[sorted_idx] + de_std[sorted_idx], 
                     alpha=0.3, color='orange', label='Ensemble ±1σ')
    
    ax2.set_xlabel('Test Engines (Sorted by true RUL)', fontsize=12)
    ax2.set_ylabel('Remaining Useful Life (cycles)', fontsize=12)
    ax2.set_title('Predictions vs Ground Truth', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = f'{OUTPUT_DIR}/uq_predictions_vs_truth.png'
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved successfully to: {output_path}")

if __name__ == '__main__':
    main()