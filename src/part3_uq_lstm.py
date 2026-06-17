import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error

# Import functions/classes from Part 2
from part2_predictive_NN import (
    RULPredictor, CMAPSSDataset, preprocess_and_window, split_train_val_engines, 
    train, BATCH_SIZE, FINAL_EPOCHS, device
)
from import_data import load_data

#==========================
# configuration
#==========================
OUTPUT_DIR = "Output/part3copy"
RUL_CAP = 125
FINAL_EPOCHS = 30
N_DE_MODELS = 3  # Number of models in the Deep Ensemble

# Use the winning configuration from Part 2
BEST_CFG = {'seq_len': 50, 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.3}


# =========================================
# MC Dropout for LSTM
# =========================================


def mc_dropout_lstm_predict(model, X_test_windows, num_passes=30, batch_size=64):
    """Generates MC Dropout predictions for LSTM."""
    print(f"Running {num_passes} stochastic forward passes for MC Dropout...")
    
    # CRITICAL: Force the model into training mode to keep Dropout active
    model.train() 
    
    # Convert numpy array to tensor and create a DataLoader for memory safety
    X_tensor = torch.FloatTensor(X_test_windows)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_passes = []
    
    with torch.no_grad():
        for _ in range(num_passes):
            pass_preds = []
            for batch_X in loader:
                batch_X = batch_X[0].to(device)
                pred = model(batch_X).cpu().numpy().flatten()
                pass_preds.extend(pred)
            all_passes.append(pass_preds)
            
    all_passes = np.array(all_passes)
    
    # Calculate Mean and Epistemic Variance
    y_mean = np.mean(all_passes, axis=0)
    y_var = np.var(all_passes, axis=0)
    
    return y_mean, y_var


#=========================================
# Ensemble of LSTMs
#=========================================


def train_lstm_ensemble(X_tr, y_tr, X_val, y_val, best_cfg, num_features, n_models=5):
    """Trains an ensemble of LSTMs using Bootstrap Sampling."""
    ensemble = []
    n_train = len(X_tr)
    
    for i in range(n_models):
        print(f"\n--- Training Ensemble Network {i+1}/{n_models} ---")
        
        # 1. Set unique seeds for random weight initialization
        seed_val = (i + 1) * 100
        torch.manual_seed(seed_val)
        np.random.seed(seed_val)
        
        # 2. Bootstrap Sampling (sample with replacement)
        indices = np.random.choice(n_train, size=n_train, replace=True)
        X_boot = X_tr[indices]
        y_boot = y_tr[indices]
        
        # 3. Create DataLoaders for this specific model
        tr_loader = DataLoader(CMAPSSDataset(X_boot, y_boot), batch_size=BATCH_SIZE, shuffle=True)
        vl_loader = DataLoader(CMAPSSDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
        
        # 4. Initialize and Train
        model = RULPredictor(num_features, 
                             hidden_size=best_cfg['hidden_size'],
                             num_layers=best_cfg['num_layers'], 
                             dropout=best_cfg['dropout']).to(device)
                             
        # Using your exact train() function from Part 2
        model, _, _ = train(model, tr_loader, vl_loader, n_epochs=FINAL_EPOCHS, verbose=False)
        ensemble.append(model)
        
    return ensemble

def ensemble_lstm_predict(ensemble, X_test_windows, batch_size=64):
    """Generates predictions across the LSTM ensemble."""
    print("Running inference across the LSTM ensemble...")
    
    X_tensor = torch.FloatTensor(X_test_windows)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_model_preds = []
    
    with torch.no_grad():
        for model in ensemble:
            model.eval() # Standard evaluation mode
            model_preds = []
            for batch_X in loader:
                batch_X = batch_X[0].to(device)
                pred = model(batch_X).cpu().numpy().flatten()
                model_preds.extend(pred)
            all_model_preds.append(model_preds)
            
    all_model_preds = np.array(all_model_preds)
    
    # Calculate Mean and Epistemic Variance
    y_mean = np.mean(all_model_preds, axis=0)
    y_var = np.var(all_model_preds, axis=0)
    
    return y_mean, y_var






# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def compute_nll(y_true, y_pred_mean, y_pred_var):
    """Computes Negative Log-Likelihood assuming a Gaussian distribution."""
    y_pred_var = np.maximum(y_pred_var, 1e-6)
    nll = 0.5 * np.log(2 * np.pi * y_pred_var) + 0.5 * ((y_true - y_pred_mean) ** 2) / y_pred_var
    return np.mean(nll)

def compute_calibration_curve(y_true, y_pred_mean, y_pred_std, confidence_levels=None):
    """Computes expected vs. observed confidence."""
    if confidence_levels is None:
        confidence_levels = np.linspace(0.05, 0.95, 19)
    
    observed_confidence = []
    for conf in confidence_levels:
        z_score = np.abs(np.percentile(np.random.randn(100000), 100 * (1 - conf) / 2))
        lower_bound = y_pred_mean - z_score * y_pred_std
        upper_bound = y_pred_mean + z_score * y_pred_std
        
        within_bounds = np.sum((y_true >= lower_bound) & (y_true <= upper_bound))
        observed_conf = within_bounds / len(y_true)
        observed_confidence.append(observed_conf)
    
    return np.array(confidence_levels), np.array(observed_confidence)


#=======================================
# plottig
#=======================================

def plot_calibration_curve(y_test, mc_mean, mc_std, mc_conf_levels, mc_obs_conf, de_mean, de_std, de_conf_levels, de_obs_conf):
    print("\nGenerating comparison plots...")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    
    # Plot A: Calibration Curves (Comparison)
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], 'k--', label='Ideal (Perfect Calibration)', linewidth=2)
    ax1.plot(mc_conf_levels, mc_obs_conf, 'o-', color='purple', label='MC Dropout', linewidth=2)
    ax1.plot(de_conf_levels, de_obs_conf, 's-', color='orange', label='Deep Ensemble', linewidth=2)
    ax1.set_xlabel('Expected Confidence (%)', fontsize=12)
    ax1.set_ylabel('Observed Confidence (%)', fontsize=12)
    ax1.set_title('Calibration Curve Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # Shared setup for Prediction Plots
    sorted_idx = np.argsort(y_test)
    test_indices = np.arange(len(y_test))
    
    # Plot B: MC Dropout Predictions
    ax2 = axes[1]
    ax2.scatter(test_indices, y_test[sorted_idx], color='black', label='Ground Truth', s=15, zorder=5)
    ax2.plot(test_indices, mc_mean[sorted_idx], '-', color='purple', label='MC Mean', linewidth=1.5)
    ax2.fill_between(test_indices, 
                     mc_mean[sorted_idx] - mc_std[sorted_idx], 
                     mc_mean[sorted_idx] + mc_std[sorted_idx], 
                     alpha=0.3, color='purple', label='MC ±1σ')
    ax2.set_xlabel('Test Engines (Sorted by true RUL)', fontsize=12)
    ax2.set_ylabel('Remaining Useful Life (cycles)', fontsize=12)
    ax2.set_title('MC Dropout: Predictions vs Ground Truth', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(alpha=0.3)

    # Plot C: Deep Ensemble Predictions
    ax3 = axes[2]
    ax3.scatter(test_indices, y_test[sorted_idx], color='black', label='Ground Truth', s=15, zorder=5)
    ax3.plot(test_indices, de_mean[sorted_idx], '-', color='orange', label='Ensemble Mean', linewidth=1.5)
    ax3.fill_between(test_indices, 
                     de_mean[sorted_idx] - de_std[sorted_idx], 
                     de_mean[sorted_idx] + de_std[sorted_idx], 
                     alpha=0.3, color='orange', label='Ensemble ±1σ')
    ax3.set_xlabel('Test Engines (Sorted by true RUL)', fontsize=12)
    ax3.set_ylabel('Remaining Useful Life (cycles)', fontsize=12)
    ax3.set_title('Deep Ensemble: Predictions vs Ground Truth', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9, loc='upper left')
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'uq_calibration_comparison.png')
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved successfully to: {output_path}")





#==========================================
# Main 
#==========================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n" + "="*70)
    print("PART 3: LSTM UNCERTAINTY QUANTIFICATION")
    print("="*70)

    # 1. Load Data using Part 2's exact preprocessing
    df_train_full, df_test = load_data()
    df_train_raw, df_val_raw = split_train_val_engines(df_train_full)
    X_tr, y_tr, X_val, y_val, X_test, num_features = preprocess_and_window(df_train_raw, df_val_raw, df_test, seq_len=BEST_CFG['seq_len'], verbose=False)
    
    try:
        rul_path = 'CMAPSSData/RUL_FD001.txt'
        if not os.path.exists(rul_path):
            rul_path = '../CMAPSSData/RUL_FD001.txt'
        rul_truth = np.minimum(
            pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'])['RUL'].values,
            RUL_CAP
        )
    except FileNotFoundError:
        print("Could not find RUL_FD001.txt.")
        exit(1)

    # ====================================================================
    # METHOD 1: MC DROPOUT
    # ====================================================================
    print("\n--- Training MC Dropout Base Model ---")
    tr_loader = DataLoader(CMAPSSDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    vl_loader = DataLoader(CMAPSSDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    
    mc_model = RULPredictor(num_features, BEST_CFG['hidden_size'], BEST_CFG['num_layers'], BEST_CFG['dropout']).to(device)
    mc_model, _, _ = train(mc_model, tr_loader, vl_loader, n_epochs=FINAL_EPOCHS, verbose=True)
    
    mc_mean, mc_var = mc_dropout_lstm_predict(mc_model, X_test, num_passes=30)
    
    # Add Aleatoric Noise
    mc_train_mean, _ = mc_dropout_lstm_predict(mc_model, X_tr, num_passes=30)
    mc_aleatoric_var = mean_squared_error(y_tr, mc_train_mean)
    mc_var = mc_var + mc_aleatoric_var
    mc_std = np.sqrt(mc_var)
    
    mc_rmse = np.sqrt(mean_squared_error(rul_truth, mc_mean))
    mc_nll = compute_nll(rul_truth, mc_mean, mc_var)

    # ====================================================================
    # METHOD 2: DEEP ENSEMBLE
    # ====================================================================
    de_ensemble = train_lstm_ensemble(X_tr, y_tr, X_val, y_val, BEST_CFG, num_features, n_models=N_DE_MODELS)
    de_mean, de_var = ensemble_lstm_predict(de_ensemble, X_test)
    
    # Add Aleatoric Noise
    de_train_mean, _ = ensemble_lstm_predict(de_ensemble, X_tr)
    de_aleatoric_var = mean_squared_error(y_tr, de_train_mean)
    de_var = de_var + de_aleatoric_var
    de_std = np.sqrt(de_var)
    
    de_rmse = np.sqrt(mean_squared_error(rul_truth, de_mean))
    de_nll = compute_nll(rul_truth, de_mean, de_var)

    # ====================================================================
    # CALIBRATION & PLOTTING
    # ====================================================================
    print("\n" + "="*70)
    print("SUMMARY COMPARISON (LSTM)")
    print("="*70)
    print(f"{'Metric':<30} {'MC Dropout':<20} {'Deep Ensemble':<20}")
    print(f"{'RMSE (cycles)':<30} {mc_rmse:<20.4f} {de_rmse:<20.4f}")
    print(f"{'Negative Log-Likelihood':<30} {mc_nll:<20.4f} {de_nll:<20.4f}")
    print(f"{'Mean Uncertainty (Std Dev)':<30} {np.mean(mc_std):<20.4f} {np.mean(de_std):<20.4f}")

    # Calculate Calibration
    mc_conf_levels, mc_obs_conf = compute_calibration_curve(rul_truth, mc_mean, mc_std)
    de_conf_levels, de_obs_conf = compute_calibration_curve(rul_truth, de_mean, de_std)
    
    # (PASTE YOUR 3-PANEL MATPLOTLIB PLOTTING CODE HERE)
    plot_calibration_curve(rul_truth, mc_mean, mc_std, mc_conf_levels, mc_obs_conf, de_mean, de_std, de_conf_levels, de_obs_conf)

if __name__ == "__main__":
    main()