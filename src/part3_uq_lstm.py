from ast import If
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
OUTPUT_DIR = "Output/part3"
RUL_CAP = 125
MC_DROPOUT_PASSES = 40
FINAL_EPOCHS = 150
EPOCHS_DE = 100 
N_DE_MODELS = 6  # Number of models in the Deep Ensemble
MAX_DE_MODELS = 12  # For DE elbow method
MAX_MC_PASSES = 100  # For MC elbow method

# The winning configuration from Part 2
BEST_CFG = {'seq_len': 50, 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.2}


# =========================================
# MC Dropout for LSTM
# =========================================


def mc_dropout_lstm_predict(model, X_test_windows, num_passes=MC_DROPOUT_PASSES, batch_size=BATCH_SIZE):
    """Generates MC Dropout predictions for LSTM."""
    print(f"Running {num_passes} stochastic forward passes for MC Dropout...")
    
    # To force the model into training mode to keep Dropout active
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


def plot_mc_dropout_elbow(mc_model, X_val, y_val, max_passes=100, step=5, MC_DROPOUT_PASSES=MC_DROPOUT_PASSES):
    """
    Evaluates the stabilization of RMSE to find the 'elbow'
    for the optimal number of MC Dropout passes.
    """
    print(f"\nCalculating MC Dropout Elbow (RMSE) up to {max_passes} passes...")
    
    m_values = np.arange(2, max_passes + 1, step)
    rmse_values = []
    
    mc_model.train()
    X_tensor = torch.FloatTensor(X_val).to(device)
    
    with torch.no_grad():
        all_passes = []
        for _ in range(max_passes):
            pred = mc_model(X_tensor).cpu().numpy().flatten()
            all_passes.append(pred)
        all_passes = np.array(all_passes)
        
    for m in m_values:
        # Get the first 'm' passes
        subset_passes = all_passes[:m, :]
        
        # Calculate the mean prediction across these passes
        subset_mean = np.mean(subset_passes, axis=0)
        
        # Calculate RMSE against the true validation labels
        rmse_at_m = np.sqrt(mean_squared_error(y_val, subset_mean))
        rmse_values.append(rmse_at_m)
        
    plt.figure(figsize=(8, 5))
    plt.plot(m_values, rmse_values, 'o-', color='purple', linewidth=2)
    plt.axvline(x=MC_DROPOUT_PASSES, color='r', linestyle='--', label=f'Current Setting (M={MC_DROPOUT_PASSES})')
    
    plt.xlabel('Number of Forward Passes (M)', fontsize=12)
    plt.ylabel('Validation RMSE (cycles)', fontsize=12)
    plt.title('MC Dropout Elbow Method: Predictive Stabilization', fontsize=13, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'mc_dropout_elbow.png')
    plt.savefig(output_path, dpi=300)
    print(f"Elbow plot saved successfully to: {output_path}")
    

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
                             
        # Using train() function from Part 2
        model, _, _ = train(model, tr_loader, vl_loader, n_epochs=EPOCHS_DE, verbose=True)
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


def plot_de_elbow(de_ensemble, X_val, y_val, current_models=N_DE_MODELS):
    """
    Evaluates the stabilization of RMSE to find the 'elbow'
    for the optimal number of Deep Ensemble models.
    """
    max_models = len(de_ensemble)
    print(f"\nCalculating Deep Ensemble Elbow (RMSE) up to {max_models} models...")
    
    b_values = np.arange(2, max_models + 1)
    rmse_values = []
    
    X_tensor = torch.FloatTensor(X_val).to(device)
    
    all_model_preds = []
    with torch.no_grad():
        for model in de_ensemble:
            model.eval()
            pred = model(X_tensor).cpu().numpy().flatten()
            all_model_preds.append(pred)
            
    all_model_preds = np.array(all_model_preds)
    
    for b in b_values:
        # Get the predictions from the first 'b' models
        subset_preds = all_model_preds[:b, :]
        
        # Calculate the ensemble mean
        subset_mean = np.mean(subset_preds, axis=0)
        
        # Calculate RMSE against the true validation labels
        rmse_at_b = np.sqrt(mean_squared_error(y_val, subset_mean))
        rmse_values.append(rmse_at_b)
        
    plt.figure(figsize=(8, 5))
    plt.plot(b_values, rmse_values, 's-', color='orange', linewidth=2)
    
    if current_models <= max_models:
        plt.axvline(x=current_models, color='r', linestyle='--', label=f'Current Setting (B={current_models})')
    
    plt.xlabel('Number of Ensemble Members (B)', fontsize=12)
    plt.ylabel('Validation RMSE (cycles)', fontsize=12)
    plt.title('Deep Ensemble Elbow Method: Predictive Stabilization', fontsize=13, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.legend()
        
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'de_elbow.png')
    plt.savefig(output_path, dpi=300)
    print(f"Deep Ensemble Elbow plot saved successfully to: {output_path}")


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

def plot_calibration_curves(mc_conf_levels, mc_obs_conf, de_conf_levels, de_obs_conf):
    print("\nGenerating Calibration Curve plot...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot([0, 1], [0, 1], 'k--', label='Ideal (Perfect Calibration)', linewidth=2)
    ax.plot(mc_conf_levels, mc_obs_conf, 'o-', color='purple', label='MC Dropout', linewidth=2)
    ax.plot(de_conf_levels, de_obs_conf, 's-', color='orange', label='Deep Ensemble', linewidth=2)
    
    ax.set_xlabel('Expected Confidence (%)', fontsize=12)
    ax.set_ylabel('Observed Confidence (%)', fontsize=12)
    ax.set_title('Calibration Curve Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'uq_calibration_curves.png')
    plt.savefig(output_path, dpi=300)
    print(f"Calibration plot saved successfully to: {output_path}")

  
def plot_uq_predictions(y_test, mc_mean, mc_std, de_mean, de_std):
    print("\nGenerating Prediction comparison plots...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Shared setup for Prediction Plots
    sorted_idx = np.argsort(y_test)
    test_indices = np.arange(len(y_test))
    
    # Plot A: MC Dropout Predictions
    ax1 = axes[0]
    ax1.scatter(test_indices, y_test[sorted_idx], color='black', label='Ground Truth', s=15, zorder=5)
    ax1.plot(test_indices, mc_mean[sorted_idx], '-', color='purple', label='MC Mean', linewidth=1.5)
    ax1.fill_between(test_indices, 
                    mc_mean[sorted_idx] - mc_std[sorted_idx], 
                    mc_mean[sorted_idx] + mc_std[sorted_idx], 
                    alpha=0.3, color='purple', label='MC ±1σ')
    ax1.set_xlabel('Test Engines (Sorted by true RUL)', fontsize=12)
    ax1.set_ylabel('Remaining Useful Life (cycles)', fontsize=12)
    ax1.set_title('MC Dropout: Predictions vs Ground Truth', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(alpha=0.3)

    # Plot B: Deep Ensemble Predictions
    ax2 = axes[1]
    ax2.scatter(test_indices, y_test[sorted_idx], color='black', label='Ground Truth', s=15, zorder=5)
    ax2.plot(test_indices, de_mean[sorted_idx], '-', color='orange', label='Ensemble Mean', linewidth=1.5)
    ax2.fill_between(test_indices, 
                    de_mean[sorted_idx] - de_std[sorted_idx], 
                    de_mean[sorted_idx] + de_std[sorted_idx], 
                    alpha=0.3, color='orange', label='Ensemble ±1σ')
    ax2.set_xlabel('Test Engines (Sorted by true RUL)', fontsize=12)
    ax2.set_ylabel('Remaining Useful Life (cycles)', fontsize=12)
    ax2.set_title('Deep Ensemble: Predictions vs Ground Truth', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'uq_predictions_comparison.png')
    plt.savefig(output_path, dpi=300)
    print(f"Predictions plot saved successfully to: {output_path}")
    

def plot_temporal_trajectory(mc_model, de_ensemble, df_val_raw, X_val, y_val, seq_len, mc_aleatoric_var, de_aleatoric_var):
    """
    Plots the full run-to-failure trajectory for a single validation engine 
    to demonstrate Prognostic Horizon and shrinking variance over time.
    """
    print("\nGenerating Temporal Trajectory plot for a single validation engine...")
    
    # 1. Isolate the first engine in the validation set
    engine_id = df_val_raw['engine'].unique()[0]
    
    # Calculate exactly how many sliding windows belong to this specific engine
    num_cycles = len(df_val_raw[df_val_raw['engine'] == engine_id])
    num_windows = num_cycles - seq_len + 1
    
    # Slice the already preprocessed validation data
    X_eng = X_val[:num_windows]
    y_eng = y_val[:num_windows]
    
    # The x-axis represents the actual flight cycles (starting after the first window)
    cycles = np.arange(seq_len, num_cycles + 1)
    
    # 2. Get MC Dropout Predictions
    mc_mean, mc_var = mc_dropout_lstm_predict(mc_model, X_eng, num_passes=MC_DROPOUT_PASSES)
    mc_std = np.sqrt(mc_var + mc_aleatoric_var)
    
    # 3. Get Deep Ensemble Predictions
    de_mean, de_var = ensemble_lstm_predict(de_ensemble, X_eng)
    de_std = np.sqrt(de_var + de_aleatoric_var)
    
    # 4. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot A: MC Dropout Temporal
    ax1 = axes[0]
    ax1.plot(cycles, y_eng, 'k-', linewidth=2.5, label='True RUL')
    ax1.plot(cycles, mc_mean, color='purple', linewidth=2, label='MC Mean Prediction')
    # Using 1.96 * std for a 95% Confidence Interval to clearly show the shrinking cone
    ax1.fill_between(cycles, 
                     mc_mean - (1.96 * mc_std), 
                     mc_mean + (1.96 * mc_std), 
                     color='purple', alpha=0.3, label='95% Confidence Interval')
    ax1.set_xlabel('Flight Cycle', fontsize=12)
    ax1.set_ylabel('Remaining Useful Life (cycles)', fontsize=12)
    ax1.set_title(f'MC Dropout: Temporal Trajectory (Engine {engine_id})', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Plot B: Deep Ensemble Temporal
    ax2 = axes[1]
    ax2.plot(cycles, y_eng, 'k-', linewidth=2.5, label='True RUL')
    ax2.plot(cycles, de_mean, color='orange', linewidth=2, label='Ensemble Mean Prediction')
    ax2.fill_between(cycles, 
                     de_mean - (1.96 * de_std), 
                     de_mean + (1.96 * de_std), 
                     color='orange', alpha=0.3, label='95% Confidence Interval')
    ax2.set_xlabel('Flight Cycle', fontsize=12)
    ax2.set_ylabel('Remaining Useful Life (cycles)', fontsize=12)
    ax2.set_title(f'Deep Ensemble: Temporal Trajectory (Engine {engine_id})', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'uq_temporal_trajectory.png')
    plt.savefig(output_path, dpi=300)
    print(f"Temporal trajectory plot saved successfully to: {output_path}")

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
    
    mc_mean, mc_var = mc_dropout_lstm_predict(mc_model, X_test, num_passes=MC_DROPOUT_PASSES)

    plot_mc_dropout_elbow(mc_model, X_val, y_val, max_passes=MAX_MC_PASSES, step=5, MC_DROPOUT_PASSES=MC_DROPOUT_PASSES)
    
    # Add Aleatoric Noise
    mc_train_mean, _ = mc_dropout_lstm_predict(mc_model, X_tr, num_passes=MC_DROPOUT_PASSES)
    mc_aleatoric_var = mean_squared_error(y_tr, mc_train_mean)
    mc_var = mc_var + mc_aleatoric_var
    mc_std = np.sqrt(mc_var)
    
    mc_rmse = np.sqrt(mean_squared_error(rul_truth, mc_mean))
    mc_nll = compute_nll(rul_truth, mc_mean, mc_var)

    # ====================================================================
    # METHOD 2: DEEP ENSEMBLE
    # ====================================================================
    GENERATE_ELBOW_PLOT = True  # Set to True to generate the Deep Ensemble elbow plot

    if GENERATE_ELBOW_PLOT:
        # 1. Train the maximum amount of models required for the elbow plot
        de_ensemble_full = train_lstm_ensemble(X_tr, y_tr, X_val, y_val, BEST_CFG, num_features, n_models=MAX_DE_MODELS)
        # 2. Generate the RMSE elbow plot using the full 15-model ensemble
        plot_de_elbow(de_ensemble_full, X_val, y_val, current_models=N_DE_MODELS)
        # 3. Slice the ensemble down to your chosen configuration for final evaluation
        de_ensemble = de_ensemble_full[:N_DE_MODELS]
    else:
        # Just train the standard 6 models to save time, skip the elbow plot
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
    
    # Plotting
    plot_calibration_curves(mc_conf_levels, mc_obs_conf, de_conf_levels, de_obs_conf)
    plot_uq_predictions(rul_truth, mc_mean, mc_std, de_mean, de_std)
    plot_temporal_trajectory(mc_model, de_ensemble, df_val_raw, X_val, y_val, BEST_CFG['seq_len'], mc_aleatoric_var, de_aleatoric_var)

if __name__ == "__main__":
    main()