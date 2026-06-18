"""
Part 2 — Data Exploration & Visualization
Generates four sets of figures that together justify:
  1. Which sensors are discarded and why (zero variance)
  2. Which remaining sensors are informative (correlation with RUL)
  3. What degradation looks like over time (sample trajectories)
  4. The overall RUL and cycle-length distributions
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from import_data import load_data

OUTPUT_DIR  = os.path.join("Output", "part2_exploration")
RUL_CAP     = 125
ZERO_VAR_THR = 1e-2  # same threshold as the NN file


# =============================================================================
# helpers
# =============================================================================

def calculate_rul(df: pd.DataFrame) -> pd.Series:
    max_cycles = df.groupby('engine')['cycle'].transform('max')
    return np.minimum(max_cycles - df['cycle'], RUL_CAP)


def feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in ['engine', 'cycle']]


def variance_split(df: pd.DataFrame):
    """Return (kept_cols, dropped_cols) based on ZERO_VAR_THR."""
    fc   = feature_cols(df)
    stds = df[fc].std()
    kept    = stds[stds >  ZERO_VAR_THR].index.tolist()
    dropped = stds[stds <= ZERO_VAR_THR].index.tolist()
    return kept, dropped, stds


# =============================================================================
# 1. Dataset statistics — printed to console
# =============================================================================

def print_dataset_stats(df_train: pd.DataFrame, df_test: pd.DataFrame):
    print("\n" + "="*60)
    print("DATASET STATISTICS — CMAPSS FD001")
    print("="*60)

    lifetimes = df_train.groupby('engine')['cycle'].max()
    rul_train = calculate_rul(df_train)

    print(f"\nTraining set")
    print(f"  Engines            : {df_train['engine'].nunique()}")
    print(f"  Total rows         : {len(df_train):,}")
    print(f"  Engine lifetime    : min={lifetimes.min()}  "
          f"mean={lifetimes.mean():.0f}  max={lifetimes.max()}  cycles")
    print(f"  RUL (capped at {RUL_CAP}) : min={rul_train.min():.0f}  "
          f"mean={rul_train.mean():.1f}  max={rul_train.max():.0f}")

    print(f"\nTest set")
    print(f"  Engines            : {df_test['engine'].nunique()}")
    print(f"  Total rows         : {len(df_test):,}")

    kept, dropped, stds = variance_split(df_train)
    print(f"\nSensor filter (std > {ZERO_VAR_THR})")
    print(f"  Total features     : {len(feature_cols(df_train))}")
    print(f"  Dropped (zero var) : {len(dropped)}  →  {dropped}")
    print(f"  Kept               : {len(kept)}")
    print("="*60 + "\n")


# =============================================================================
# 2. Sensor variance — bar chart showing which sensors are dropped
# =============================================================================

def plot_sensor_variance(df_train: pd.DataFrame):
    kept, dropped, stds = variance_split(df_train)
    stds_sorted = stds.sort_values(ascending=True)

    colors = ['#E84A5F' if c in dropped else '#4A88E8'
              for c in stds_sorted.index]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(stds_sorted.index, stds_sorted.values, color=colors)
    ax.axvline(ZERO_VAR_THR, color='white', linestyle='--', linewidth=1.2,
               label=f'Zero-variance threshold ({ZERO_VAR_THR})')
    ax.set_xlabel('Standard deviation (raw units)', fontsize=12)
    ax.set_title('Sensor standard deviation — red bars are discarded', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, linestyle='--', axis='x')

    # annotate dropped sensors
    for bar, col in zip(bars, stds_sorted.index):
        if col in dropped:
            ax.text(bar.get_width() + stds_sorted.max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    'discarded', va='center', fontsize=8, color='#E84A5F')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'sensor_variance.png')
    plt.savefig(path, dpi=200)
    plt.show()
    print(f"Saved: {path}")


# =============================================================================
# 3. Sensor–RUL correlation — bar chart for kept sensors
# =============================================================================

def plot_sensor_rul_correlation(df_train: pd.DataFrame):
    kept, _, _ = variance_split(df_train)
    rul = calculate_rul(df_train)

    corrs = df_train[kept].corrwith(rul).sort_values()
    colors = ['#E84A5F' if v < 0 else '#4A88E8' for v in corrs.values]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(corrs.index, corrs.values, color=colors)
    ax.axvline(0, color='white', linewidth=0.8)
    ax.set_xlabel('Pearson correlation with RUL', fontsize=12)
    ax.set_title('Sensor correlation with RUL (kept sensors only)', fontsize=14)
    ax.grid(True, alpha=0.2, linestyle='--', axis='x')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'sensor_rul_correlation.png')
    plt.savefig(path, dpi=200)
    plt.show()
    print(f"Saved: {path}")

    # print top positive and negative correlators
    print("Top 5 positively correlated with RUL:")
    print(corrs.tail(5).to_string())
    print("\nTop 5 negatively correlated with RUL:")
    print(corrs.head(5).to_string())


# =============================================================================
# 4. Degradation trajectories — top-4 most correlated sensors, 5 engines
# =============================================================================

def plot_degradation_trajectories(df_train: pd.DataFrame):
    kept, _, _ = variance_split(df_train)
    rul = calculate_rul(df_train)

    # pick 4 sensors with highest |correlation| with RUL
    corrs = df_train[kept].corrwith(rul).abs().sort_values(ascending=False)
    top_sensors = corrs.head(4).index.tolist()

    # pick 5 engines with the longest lifetimes for a clear visual
    lifetimes   = df_train.groupby('engine')['cycle'].max().sort_values(ascending=False)
    sample_engines = lifetimes.head(5).index.tolist()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    colors = ['#4A88E8', '#E84A5F', '#2ECC71', '#F39C12', '#9B59B6']

    for ax, sensor in zip(axes, top_sensors):
        for eng, col in zip(sample_engines, colors):
            mask = df_train['engine'] == eng
            cycles = df_train.loc[mask, 'cycle'].values
            vals   = df_train.loc[mask, sensor].values
            ax.plot(cycles, vals, color=col, alpha=0.8, linewidth=1.2,
                    label=f'Engine {eng}')
        ax.set_title(f'{sensor}  (|r| = {corrs[sensor]:.3f})', fontsize=11)
        ax.set_xlabel('Cycle', fontsize=10)
        ax.set_ylabel('Sensor value', fontsize=10)
        ax.grid(True, alpha=0.2, linestyle='--')

    axes[0].legend(fontsize=8, loc='best')
    fig.suptitle('Degradation trajectories — top-4 sensors by |correlation| with RUL',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'degradation_trajectories.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"Saved: {path}")


# =============================================================================
# 5. RUL and lifetime distributions
# =============================================================================

def plot_distributions(df_train: pd.DataFrame):
    lifetimes = df_train.groupby('engine')['cycle'].max()
    rul_at_last = lifetimes.copy()   # RUL before cap = lifetime itself; after cap = min(lifetime, 125)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(lifetimes.values, bins=20, color='#4A88E8', edgecolor='white', linewidth=0.5)
    ax1.set_xlabel('Engine lifetime (cycles)', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Distribution of engine lifetimes', fontsize=12)
    ax1.axvline(lifetimes.mean(), color='#E84A5F', linestyle='--',
                label=f'Mean = {lifetimes.mean():.0f}')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2, linestyle='--')

    rul_all = calculate_rul(df_train)
    ax2.hist(rul_all.values, bins=30, color='#2ECC71', edgecolor='white', linewidth=0.5)
    ax2.set_xlabel(f'RUL (capped at {RUL_CAP})', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Distribution of RUL labels (all training rows)', fontsize=12)
    ax2.axvline(RUL_CAP, color='#E84A5F', linestyle='--',
                label=f'Cap = {RUL_CAP}')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2, linestyle='--')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'rul_lifetime_distributions.png')
    plt.savefig(path, dpi=200)
    plt.show()
    print(f"Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_train, df_test = load_data()

    print_dataset_stats(df_train, df_test)
    plot_sensor_variance(df_train)
    plot_sensor_rul_correlation(df_train)
    plot_degradation_trajectories(df_train)
    plot_distributions(df_train)

    print(f"\nAll figures saved to {OUTPUT_DIR}/")