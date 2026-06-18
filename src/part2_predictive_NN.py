import os
import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from part2_predictive import train_random_forest
import matplotlib.pyplot as plt
import time
from import_data import load_data

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR   = "Output/part2_direct_lstm"
RUL_CAP      = 125
BATCH_SIZE   = 64
TUNE_EPOCHS  = 25    # epochs per candidate during search
FINAL_EPOCHS = 150   # epochs for the winning config
LR           = 1e-3
TRAIN_RATIO  = 0.8

# Grid to search — 24 combinations
SEARCH_SPACE = [
    {'seq_len': sl, 'hidden_size': hs, 'num_layers': nl, 'dropout': dr}
    for sl in [20, 30, 50]
    for hs in [64, 128]
    for nl in [1, 2]
    for dr in [0.2, 0.3]
]

# This was added after hyperparameter tuning as the the best config had signs of overfitting (train RMSE much lower than val RMSE).
FORCE_CFG = {'seq_len': 50, 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.2}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
# =============================================================================
# 1. DATA PREPARATION
# =============================================================================

def calculate_rul(df: pd.DataFrame) -> np.ndarray:
    max_cycles = df.groupby('engine')['cycle'].transform('max')
    rul = max_cycles - df['cycle']
    return np.minimum(rul, RUL_CAP).values


def split_train_val_engines(df: pd.DataFrame, train_ratio=TRAIN_RATIO, seed=42):
    """Engine-level split so no engine appears in both train and val."""
    rng = np.random.default_rng(seed)
    unique_engines = df['engine'].unique()
    rng.shuffle(unique_engines)
    split_idx = int(len(unique_engines) * train_ratio)
    train_engines = unique_engines[:split_idx]
    df_train = df[df['engine'].isin(train_engines)].copy()
    df_val   = df[~df['engine'].isin(train_engines)].copy()
    return df_train, df_val


def preprocess_and_window(df_train: pd.DataFrame,
                           df_val:   pd.DataFrame,
                           df_test:  pd.DataFrame,
                           seq_len:  int,
                           verbose:  bool = True):
    """
    1. Drop zero-variance sensors 
    2. StandardScaler fit on train only.
    3. Build sliding-window sequences of length seq_len.
    4. For test: take the last seq_len cycles per engine.
    """
    feature_cols = [c for c in df_train.columns if c not in ['engine', 'cycle']]
    stds = df_train[feature_cols].std()
    cols_to_keep = stds[stds > 1e-2].index.tolist()
    cols_dropped  = stds[stds <= 1e-2].index.tolist()

    if verbose:
        print(f"  Dropped zero-variance sensors ({len(cols_dropped)}): {cols_dropped}")
        print(f"  Keeping {len(cols_to_keep)} features after variance filter.")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(df_train[cols_to_keep])
    val_scaled   = scaler.transform(df_val[cols_to_keep])
    test_scaled  = scaler.transform(df_test[cols_to_keep])

    def build_windows(df_source, scaled_data):
        rul     = calculate_rul(df_source)
        engines = df_source['engine'].values
        X_seq, y_seq = [], []
        for eid in np.unique(engines):
            mask  = engines == eid
            edata = scaled_data[mask]
            elbl  = rul[mask]
            for i in range(len(edata) - seq_len + 1):
                X_seq.append(edata[i : i + seq_len])
                y_seq.append(elbl[i + seq_len - 1])
        return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)

    X_tr,  y_tr  = build_windows(df_train, train_scaled)
    X_val, y_val = build_windows(df_val,   val_scaled)

    # Test: last seq_len cycles per engine
    X_test = []
    for eid in np.unique(df_test['engine'].values):
        mask  = df_test['engine'].values == eid
        edata = test_scaled[mask]
        if len(edata) >= seq_len:
            X_test.append(edata[-seq_len:])
        else:
            pad = seq_len - len(edata)
            X_test.append(np.pad(edata, ((pad, 0), (0, 0)), mode='edge'))
    X_test = np.array(X_test, dtype=np.float32)

    num_features = X_tr.shape[2]
    if verbose:
        print(f"  Train windows: {X_tr.shape}   Val windows: {X_val.shape}   "
              f"Test windows: {X_test.shape}")
    return X_tr, y_tr, X_val, y_val, X_test, num_features


# =============================================================================
# 2. DATASET
# =============================================================================

class CMAPSSDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(1)

    def __len__(self):            return len(self.X)
    def __getitem__(self, i):     return self.X[i], self.y[i]


# =============================================================================
# 3. MODEL
# =============================================================================

class RULPredictor(nn.Module):
    def __init__(self, num_features, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(num_features, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=lstm_dropout)
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(hidden_size, 32)
        self.fc2  = nn.Linear(32, 16)
        self.fc3  = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]          # last time-step hidden state
        out = self.drop(out)
        out = torch.relu(self.fc1(out))
        out = self.drop(out)
        out = torch.relu(self.fc2(out))
        return self.fc3(out)         # linear output — no activation


# =============================================================================
# 4. TRAINING
# =============================================================================

def train(model, train_loader, val_loader, n_epochs=FINAL_EPOCHS, verbose=True):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)
    
    

    best_val_rmse = float('inf')
    best_weights  = None
    history       = {'train_rmse': [], 'val_rmse': []}

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n{'='*55}")
        print(f"TRAINING  —  LSTM  ({n_params:,} params)  device={device}")
        print(f"{'='*55}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(X)
        train_rmse = np.sqrt(train_loss / len(train_loader.dataset))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_loss += criterion(model(X), y).item() * len(X)
        val_rmse = np.sqrt(val_loss / len(val_loader.dataset))

        scheduler.step(val_rmse)
        history['train_rmse'].append(train_rmse)
        history['val_rmse'].append(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_weights  = copy.deepcopy(model.state_dict())

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"Epoch {epoch:03d}/{n_epochs}  |  "
                  f"Train RMSE: {train_rmse:6.2f}  |  "
                  f"Val RMSE: {val_rmse:6.2f}  |  "
                  f"Best Val: {best_val_rmse:6.2f}")

    model.load_state_dict(best_weights)
    if verbose:
        print(f"\nBest val RMSE: {best_val_rmse:.2f}")
    return model, history, best_val_rmse


# =============================================================================
# 5. HYPERPARAMETER TUNING
# =============================================================================

def tune_hyperparameters(df_train_raw, df_val_raw, df_test_raw):
    """
    Grid search over SEARCH_SPACE.
    Each configuration is trained for TUNE_EPOCHS and evaluated on val RMSE.
    The winner is returned as a dict and the full results are saved to CSV.
    """
    print(f"\nHyperparameter search: {len(SEARCH_SPACE)} configs × {TUNE_EPOCHS} epochs")
    print("=" * 65)

    results = []
    for i, cfg in enumerate(SEARCH_SPACE, 1):
        X_tr, y_tr, X_val, y_val, _, n_feat = preprocess_and_window(
            df_train_raw, df_val_raw, df_test_raw,
            seq_len=cfg['seq_len'], verbose=False
        )
        tr_loader = DataLoader(CMAPSSDataset(X_tr,  y_tr),  batch_size=BATCH_SIZE, shuffle=True)
        vl_loader = DataLoader(CMAPSSDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

        # Train for TUNE_EPOCHS and record the best val RMSE achieved during training
        model = RULPredictor(n_feat, cfg['hidden_size'],cfg['num_layers'], cfg['dropout']).to(device)

        # Train and get the best val RMSE for this configuration
        _, history, best_val = train(model, tr_loader, vl_loader,
                                     n_epochs=TUNE_EPOCHS, verbose=False)

        # Final train RMSE (last epoch) — gap vs val reveals overfitting
        final_train = history['train_rmse'][-1]
        gap = best_val - final_train
        
        # Save results for this configuration
        results.append({**cfg,
                        'train_rmse': round(final_train, 3),
                        'val_rmse':   round(best_val, 3),
                        'gap':        round(gap, 3)})
        print(f"  [{i:02d}/{len(SEARCH_SPACE)}]  seq={cfg['seq_len']:2d}  "
              f"hidden={cfg['hidden_size']:3d}  layers={cfg['num_layers']}  "
              f"dropout={cfg['dropout']}  →  "
              f"train {final_train:.2f}  val {best_val:.2f}  gap {gap:.2f}")

    df_res = pd.DataFrame(results).sort_values('val_rmse').reset_index(drop=True)
    df_res.to_csv(os.path.join(OUTPUT_DIR, 'hparam_search.csv'), index=False)

    # Return the best configuration
    best = df_res.iloc[0].to_dict()
    best_cfg = {
        'seq_len':     int(best['seq_len']),
        'hidden_size': int(best['hidden_size']),
        'num_layers':  int(best['num_layers']),
        'dropout':     float(best['dropout']),
    }
    print(f"\nBest config: {best_cfg}  (val RMSE {best['val_rmse']:.2f})")

    if FORCE_CFG is not None:
        print(f"\nOverriding grid search winner with FORCE_CFG: {FORCE_CFG}")
        best_cfg = FORCE_CFG
    return best_cfg


# =============================================================================
# 6. EVALUATION & PLOTS
# =============================================================================
def evaluate(model, X_test: np.ndarray, rul_truth: np.ndarray):
    model.eval()
    # Predict and clip to [0, RUL_CAP]
    with torch.no_grad():
        preds = model(torch.from_numpy(X_test).to(device))
        preds = np.clip(preds.cpu().numpy().flatten(), 0, RUL_CAP)

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(rul_truth, preds))
    mae  = mean_absolute_error(rul_truth, preds)
    print(f"\n{'='*55}")
    print("TEST SET RESULTS  (100 engines, FD001)")
    print(f"{'='*55}")
    print(f"  RMSE : {rmse:.3f}")
    print(f"  MAE  : {mae:.3f}")
    print(f"{'='*55}")
    return preds, rmse, mae


def plot_training_curves(history: dict):
    epochs = range(1, len(history['train_rmse']) + 1)
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history['train_rmse'], label='Train RMSE', color='#4A88E8')
    plt.plot(epochs, history['val_rmse'],   label='Val RMSE',   color='#E84A5F')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (cycles)')
    plt.title('LSTM — Training vs Validation RMSE')
    plt.legend()
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=200)
    # plt.show()


def plot_rul_predictions(rul_truth: np.ndarray, preds: np.ndarray):
    sort_idx = np.argsort(rul_truth)
    plt.figure(figsize=(14, 6))
    plt.plot(rul_truth[sort_idx], label='True RUL', color='black', linewidth=2.5, zorder=2)
    plt.scatter(range(len(preds)), preds[sort_idx],
                label='Predicted RUL', color='#4A88E8', alpha=0.8, s=35, zorder=3)
    plt.axhline(RUL_CAP, color='#E84A5F', linestyle='--', alpha=0.6,
                label=f'RUL Cap ({RUL_CAP})')
    plt.title('CMAPSS FD001 — Actual vs Predicted Final RUL (LSTM)', fontsize=16)
    plt.xlabel('Engine (sorted by True RUL)', fontsize=12)
    plt.ylabel('Remaining Useful Life (cycles)', fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rul_predictions.png'), dpi=300)
    # plt.show()


def plot_hparam_search(csv_path: str):
    df = pd.read_csv(csv_path).sort_values('val_rmse')
    labels = [f"seq={r.seq_len} h={r.hidden_size} l={r.num_layers} d={r.dropout}"
              for _, r in df.iterrows()]
    plt.figure(figsize=(12, 6))
    plt.barh(labels[::-1], df['val_rmse'].values[::-1], color='#4A88E8')
    plt.xlabel('Best Val RMSE (cycles)')
    plt.title(f'Hyperparameter Search — {len(df)} configs × {TUNE_EPOCHS} epochs')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hparam_search.png'), dpi=200)
    # plt.show()


def plot_model_comparison(rul_truth: np.ndarray, preds: np.ndarray, rf_preds: np.ndarray,
                           rmse: float, mae: float, rf_rmse: float, rf_mae: float):
    """Side-by-side prediction overlay and RMSE/MAE bar chart, LSTM vs Random Forest."""
    sort_idx = np.argsort(rul_truth)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(rul_truth[sort_idx], color='black', linewidth=2.5, label='True RUL', zorder=3)
    ax.scatter(range(len(preds)),    preds[sort_idx],    color='#4A88E8', s=30, alpha=0.7,
               label=f'LSTM  (RMSE={rmse:.1f})', zorder=2)
    ax.scatter(range(len(rf_preds)), rf_preds[sort_idx], color='#E84A5F', s=30, alpha=0.7,
               marker='x', label=f'RF    (RMSE={rf_rmse:.1f})', zorder=2)
    ax.axhline(RUL_CAP, color='grey', linestyle='--', alpha=0.5, label=f'RUL cap ({RUL_CAP})')
    ax.set_xlabel('Engine (sorted by True RUL)')
    ax.set_ylabel('RUL (cycles)')
    ax.set_title('LSTM vs Random Forest — Test Predictions')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, linestyle='--')

    ax = axes[1]
    labels  = ['RMSE', 'MAE']
    lstm_v  = [rmse,    mae]
    rf_v    = [rf_rmse, rf_mae]
    x, w    = np.arange(2), 0.35
    ax.bar(x - w/2, lstm_v, w, label='LSTM',          color='#4A88E8')
    ax.bar(x + w/2, rf_v,   w, label='Random Forest', color='#E84A5F')
    for i, (l, r) in enumerate(zip(lstm_v, rf_v)):
        ax.text(i - w/2, l + 0.2, f'{l:.1f}', ha='center', fontsize=9)
        ax.text(i + w/2, r + 0.2, f'{r:.1f}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Error (cycles)')
    ax.set_title('Test Set Metric Comparison')
    ax.legend(); ax.grid(True, alpha=0.2, linestyle='--', axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=200)
    # plt.show()


# =============================================================================
# 7. RANDOM FOREST BASELINE
# =============================================================================

def run_random_forest_baseline(X_tr, y_tr, X_val, y_val, X_test, rul_truth):
    """
    Random Forest baseline (same preprocessing, last timestep only).
    Used as a comparison and is imported from predictive.py to avoid code duplication.
    """
    print(f"\n{'='*55}")
    print("RANDOM FOREST BASELINE  (last timestep of each window)")
    print(f"{'='*55}")
    X_tr_flat   = X_tr[:, -1, :]    # (n_windows, 14) — current sensor reading
    X_val_flat  = X_val[:, -1, :]
    X_test_flat = X_test[:, -1, :]

    rf_start = time.time()
    rf_model, _ = train_random_forest(X_tr_flat, y_tr, X_val_flat, y_val)
    rf_preds     = np.clip(rf_model.predict(X_test_flat), 0, RUL_CAP)
    rf_rmse      = np.sqrt(mean_squared_error(rul_truth, rf_preds))
    rf_mae       = mean_absolute_error(rul_truth, rf_preds)
    rf_time      = time.time() - rf_start
    print(f"\nRandom Forest — Test RMSE: {rf_rmse:.3f}   MAE: {rf_mae:.3f}")

    return rf_model, rf_preds, rf_rmse, rf_mae, rf_time


# =============================================================================
# 8. MAIN
# =============================================================================

def load_rul_truth():
    try:
        rul_path = 'CMAPSSData/RUL_FD001.txt'
        if not os.path.exists(rul_path):
            rul_path = '../CMAPSSData/RUL_FD001.txt'
        return np.minimum(
            pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'])['RUL'].values,
            RUL_CAP
        )
    except FileNotFoundError:
        print("Could not find RUL_FD001.txt.")
        exit(1)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_train_full, df_test = load_data()
    df_train_raw, df_val_raw = split_train_val_engines(df_train_full)
    print(f"Train engines: {df_train_raw['engine'].nunique()}  "
          f"Val engines: {df_val_raw['engine'].nunique()}  "
          f"Test engines: {df_test['engine'].nunique()}")

    #Step 1: hyperparameter search
    best_cfg = tune_hyperparameters(df_train_raw, df_val_raw, df_test)

    #Step 2: full training with best configuratoin
    print(f"\n{'='*55}")
    print(f"FINAL TRAINING  with best config: {best_cfg}")
    print(f"{'='*55}")

    # Re-prepare data with the best seq_len and all engines (train+val) for final training
    X_tr, y_tr, X_val, y_val, X_test, num_features = preprocess_and_window(
        df_train_raw, df_val_raw, df_test, seq_len=best_cfg['seq_len'], verbose=True
    )
    train_loader = DataLoader(CMAPSSDataset(X_tr,  y_tr),  batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(CMAPSSDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = RULPredictor(num_features, best_cfg['hidden_size'],
                         best_cfg['num_layers'], best_cfg['dropout']).to(device)

    # Start timer for comparison with Random Forest baseline
    lstm_start = time.time()
    model, history, _ = train(model, train_loader, val_loader,
                               n_epochs=FINAL_EPOCHS, verbose=True)
    lstm_time = time.time() - lstm_start

    plot_training_curves(history)
    plot_hparam_search(os.path.join(OUTPUT_DIR, 'hparam_search.csv'))

    # Step 3: test evaluation
    rul_truth = load_rul_truth()
    preds, rmse, mae = evaluate(model, X_test, rul_truth)
    plot_rul_predictions(rul_truth, preds)

    # Step 4: Random Forest baseline for comparison
    rf_model, rf_preds, rf_rmse, rf_mae, rf_time = run_random_forest_baseline(
        X_tr, y_tr, X_val, y_val, X_test, rul_truth
    )

    # Step 5: comparison plot & summary table
    plot_model_comparison(rul_truth, preds, rf_preds, rmse, mae, rf_rmse, rf_mae)

    summary = pd.DataFrame([
        {**best_cfg, 'model': 'LSTM',          'test_rmse': rmse,    'test_mae': mae},
        {            'model': 'Random Forest',  'test_rmse': rf_rmse, 'test_mae': rf_mae},
    ])
    summary.to_csv(os.path.join(OUTPUT_DIR, 'final_results.csv'), index=False)
    print(f"\nResults saved to {OUTPUT_DIR}/")
    print(f"\nTraining time — LSTM: {lstm_time:.1f}s   RF: {rf_time:.1f}s")


if __name__ == "__main__":
    main()
