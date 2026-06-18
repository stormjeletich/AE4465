# =============================================================================
# PART 2: PREDICTIVE MAINTENANCE - RUL PREDICTION MODEL
# =============================================================================
# 
# This module implements a machine learning-based Remaining Useful Life (RUL)
# prediction system for aircraft engine prognostics using the C-MAPSS dataset.
#
# WORKFLOW OVERVIEW:
# 1. Data Loading & Exploration: Load FD001 training/test data and analyze
# 2. Data Preprocessing: Normalize features, handle outliers, engineer features
# 3. RUL Labeling: Calculate RUL for training data and apply 125-cycle cap
# 4. Data Splitting: Split by engine to prevent data leakage
# 5. Model Selection & Training: Train multiple ML models with hyperparameter tuning
# 6. Evaluation: Assess performance on validation set
# 7. Prediction: Make RUL predictions on test set
# 8. Results Visualization: Plot performance metrics and comparisons
#
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Import data loading function
from import_data import load_data

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Output', 'part2'))
RUL_CAP = 125  # Cap RUL at 125 cycles (healthy engines are difficult to predict)
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Sensor and operational condition columns to use for modeling
#  We exclude 'engine' and 'cycle' columns as per instructions
# (they would leak information about RUL in this dataset)
FEATURE_COLUMNS_TO_EXCLUDE = ['engine', 'cycle']

# Random seed for reproducibility
np.random.seed(RANDOM_STATE)


# =============================================================================
# SECTION 1: DATA EXPLORATION & ANALYSIS
# =============================================================================

def explore_data(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    """
    Perform initial data exploration and analysis.
    
    This section addresses requirement: 
    "Describe the data in this dataset. Based on this initial data exploration,
    are there any sensors you would disregard?"
    
    Analysis includes:
    - Shape and missing values
    - Statistical summary (mean, std, min, max)
    - Feature distributions and correlations
    - Potential candidates for removal
    """
    print("\n" + "="*80)
    print("SECTION 1: DATA EXPLORATION & ANALYSIS")
    print("="*80)
    
    print("\n1.1 Dataset Shape and Structure")
    print(f"Training set shape: {df_train.shape}")
    print(f"Test set shape: {df_test.shape}")
    print(f"Number of unique engines in training: {df_train['engine'].nunique()}")
    print(f"Number of unique engines in test: {df_test['engine'].nunique()}")
    
    # Check for missing values
    print("\n1.2 Missing Values")
    missing_train = df_train.isnull().sum().sum()
    missing_test = df_test.isnull().sum().sum()
    print(f"Missing values in training set: {missing_train}")
    print(f"Missing values in test set: {missing_test}")
    
    # Statistical summary
    feature_cols = [col for col in df_train.columns if col not in FEATURE_COLUMNS_TO_EXCLUDE]
    print("\n1.3 Feature Statistics (Training Set)")
    print(df_train[feature_cols].describe())
    
    # Analyze feature variance and distribution
    print("\n1.4 Feature Variance Analysis")
    variances = df_train[feature_cols].var()
    std_devs = df_train[feature_cols].std()
    
    # Identify features with near-zero variance (candidates for removal)
    zero_var_threshold = 0.01
    near_zero_vars = variances[variances < zero_var_threshold]
    
    if len(near_zero_vars) > 0:
        print(f"\nFeatures with very low variance (candidates for removal):")
        print(near_zero_vars)
    else:
        print("\nNo features with near-zero variance detected.")
    
    # Correlation analysis
    print("\n1.5 Feature Correlations (Top absolute correlations)")
    corr_matrix = df_train[feature_cols].corr().abs()
    
    # Find highly correlated features
    print("\nHighly correlated feature pairs (correlation > 0.95):")
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.95:
                print(f"  {corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.4f}")
    
    # Create visualization
    plt.figure(figsize=(14, 10))
    correlation_subset = df_train[feature_cols].corr()
    sns.heatmap(correlation_subset, cmap='coolwarm', center=0, 
                xticklabels=True, yticklabels=True, cbar_kws={'label': 'Correlation'})
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_feature_correlations.png', dpi=300)
    plt.close()
    print(f"\nCorrelation plot saved to '{OUTPUT_DIR}/01_feature_correlations.png'")


# =============================================================================
# SECTION 2: DATA PREPROCESSING
# =============================================================================

def calculate_rul(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Remaining Useful Life (RUL) for each cycle of each engine.
    
    RUL is defined as: RUL = (max_cycle_for_engine - current_cycle)
    
    The RUL is capped at 125 cycles because:
    - Healthy engines are difficult to predict
    - RUL > 125 indicates a healthy engine
    - Capping helps the model focus on degradation patterns
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'engine' and 'cycle' columns
    
    Returns:
    --------
    pd.Series
        RUL for each row, capped at RUL_CAP
    """
    # Group by engine and calculate max cycle (end-of-life)
    max_cycles = df.groupby('engine')['cycle'].transform('max')
    
    # RUL = cycles remaining until failure
    rul = max_cycles - df['cycle']
    
    # Cap RUL at 125 cycles
    rul = np.minimum(rul, RUL_CAP)
    
    return rul


def preprocess_data(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Preprocess training and test data for machine learning.
    
    Steps:
    1. Remove engine and cycle columns (as per instructions to prevent information leakage)
    2. Remove any constant-variance features
    3. Standardize all features (zero mean, unit variance)
    
    Parameters:
    -----------
    df_train : pd.DataFrame
        Training data
    df_test : pd.DataFrame
        Test data
    
    Returns:
    --------
    df_train_processed : pd.DataFrame
        Preprocessed training data with scaled features
    df_test_processed : pd.DataFrame
        Preprocessed test data with scaled features
    scaler : StandardScaler
        Fitted scaler object (for reference/reproducibility)
    """
    print("\n" + "="*80)
    print("SECTION 2: DATA PREPROCESSING")
    print("="*80)
    
    # Get feature columns (exclude index columns)
    feature_cols = [col for col in df_train.columns if col not in FEATURE_COLUMNS_TO_EXCLUDE]
    print(f"\n2.1 Feature Selection")
    print(f"Total features available: {len(feature_cols)}")
    print(f"Features: {feature_cols}")
    
    # Create a copy and extract features
    df_train_processed = df_train[feature_cols].copy()
    df_test_processed = df_test[feature_cols].copy()
    
    # Remove constant-variance features (if any)
    print("\n2.2 Removing Low-Variance Features")
    variances = df_train_processed.var()
    low_var_threshold = 0.01
    cols_to_keep = variances[variances > low_var_threshold].index.tolist()
    
    removed_cols = [col for col in feature_cols if col not in cols_to_keep]
    if removed_cols:
        print(f"Removing features with variance < {low_var_threshold}: {removed_cols}")
        df_train_processed = df_train_processed[cols_to_keep]
        df_test_processed = df_test_processed[cols_to_keep]
    else:
        print("No low-variance features detected.")
    
    # Standardization: zero mean, unit variance
    print("\n2.3 Feature Scaling (Standardization)")
    scaler = StandardScaler()
    df_train_processed = pd.DataFrame(
        scaler.fit_transform(df_train_processed),
        columns=df_train_processed.columns,
        index=df_train_processed.index
    )
    df_test_processed = pd.DataFrame(
        scaler.transform(df_test_processed),
        columns=df_test_processed.columns,
        index=df_test_processed.index
    )
    print(f"Scaling complete. Features standardized to mean=0, std=1")
    
    return df_train_processed, df_test_processed, scaler


# =============================================================================
# SECTION 3: RUL LABELING & DATA SPLITTING
# =============================================================================

def prepare_training_data(df_train: pd.DataFrame, df_train_features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare training data with RUL labels.
    
    Steps:
    1. Calculate RUL for each cycle
    2. Create engine indices to track engine membership
    3. Convert to numpy arrays for model training
    
    Parameters:
    -----------
    df_train : pd.DataFrame
        Original training data with engine and cycle info
    df_train_features : pd.DataFrame
        Preprocessed feature data
    
    Returns:
    --------
    X_train : np.ndarray
        Feature matrix (n_samples, n_features)
    y_train : np.ndarray
        RUL labels (n_samples,)
    engine_ids : np.ndarray
        Engine IDs for each sample (used for splitting)
    """
    # Calculate RUL
    y_train = calculate_rul(df_train).values
    
    # Extract engine IDs
    engine_ids = df_train['engine'].values
    
    # Features
    X_train = df_train_features.values
    
    return X_train, y_train, engine_ids


def split_data_by_engine(X: np.ndarray, y: np.ndarray, engine_ids: np.ndarray, 
                        train_ratio: float = 0.6, val_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets while ensuring engines are not mixed.
    
    CRITICAL: To prevent data leakage, we must ensure that the same engine does not
    appear in both training and validation/test sets. This is important because:
    - Engines have unique degradation patterns
    - Models might memorize engine-specific signatures
    - In practice, we would not have access to the same engine in validation/test
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        RUL labels
    engine_ids : np.ndarray
        Engine IDs
    train_ratio : float
        Fraction of engines for training (default 0.6)
    val_ratio : float
        Fraction of engines for validation (default 0.2)
    
    Returns:
    --------
    X_train, y_train : Training features and labels
    X_val, y_val : Validation features and labels
    X_test, y_test : Test features and labels
    """
    print("\n" + "="*80)
    print("SECTION 3: DATA SPLITTING (BY ENGINE)")
    print("="*80)
    
    # Get unique engines and shuffle them
    unique_engines = np.unique(engine_ids)
    n_engines = len(unique_engines)
    shuffled_engines = np.random.permutation(unique_engines)
    
    # Calculate split sizes
    n_train = int(n_engines * train_ratio)
    n_val = int(n_engines * val_ratio)
    
    train_engines = shuffled_engines[:n_train]
    val_engines = shuffled_engines[n_train:n_train+n_val]
    test_engines = shuffled_engines[n_train+n_val:]
    
    # Split data by engine membership
    train_mask = np.isin(engine_ids, train_engines)
    val_mask = np.isin(engine_ids, val_engines)
    test_mask = np.isin(engine_ids, test_engines)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"\n3.1 Train/Val/Test Split (by Engine)")
    print(f"Total engines: {n_engines}")
    print(f"Training engines: {len(train_engines)} ({100*len(train_engines)/n_engines:.1f}%)")
    print(f"Validation engines: {len(val_engines)} ({100*len(val_engines)/n_engines:.1f}%)")
    print(f"Test engines: {len(test_engines)} ({100*len(test_engines)/n_engines:.1f}%)")
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# =============================================================================
# SECTION 4: MODEL TRAINING & HYPERPARAMETER TUNING
# =============================================================================

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, 
                        X_val: np.ndarray, y_val: np.ndarray) -> Tuple[RandomForestRegressor, Dict[str, float]]:
    """
    Train a Random Forest regressor with hyperparameter tuning.
    
    Random Forest is chosen because:
    - Non-parametric: no assumptions about feature distributions
    - Handles non-linear relationships well
    - Robust to outliers
    - Provides feature importance rankings
    - Good for high-dimensional data
    
    Hyperparameters tuned:
    - n_estimators: number of trees (more trees = better but slower)
    - max_depth: maximum tree depth (controls complexity/overfitting)
    - min_samples_split: minimum samples required to split (regularization)
    - min_samples_leaf: minimum samples in leaf (regularization)
    """
    print("\n" + "="*80)
    print("SECTION 4A: RANDOM FOREST MODEL")
    print("="*80)
    
    print("\n4A.1 Hyperparameter Grid Search")
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create base estimator
    rf_base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    
    # Grid search with cross-validation on training set
    gs_rf = GridSearchCV(rf_base, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    gs_rf.fit(X_train, y_train)
    
    best_rf = gs_rf.best_estimator_
    print(f"Best hyperparameters: {gs_rf.best_params_}")
    
    # Evaluate on validation set
    y_val_pred = best_rf.predict(X_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
    mae_val = mean_absolute_error(y_val, y_val_pred)
    r2_val = r2_score(y_val, y_val_pred)
    
    print(f"\nValidation Performance:")
    print(f"  RMSE: {rmse_val:.3f} cycles")
    print(f"  MAE:  {mae_val:.3f} cycles")
    print(f"  R²:   {r2_val:.3f}")
    
    metrics = {'RMSE': rmse_val, 'MAE': mae_val, 'R2': r2_val}
    
    return best_rf, metrics


# =============================================================================
# SECTION 5: MODEL EVALUATION & COMPARISON
# =============================================================================

def evaluate_models(models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """
    Evaluate all trained models on the held-out test set.
    
    Parameters:
    -----------
    models : Dict[str, Any]
        Dictionary of model names and fitted models
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test RUL labels
    
    Returns:
    --------
    results_df : pd.DataFrame
        DataFrame with test set metrics for each model
    """
    print("\n" + "="*80)
    print("SECTION 5: MODEL EVALUATION (TEST SET)")
    print("="*80)
    
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'RMSE (cycles)': rmse,
            'MAE (cycles)': mae,
            'R² Score': r2
        })
        
        print(f"\n{name}:")
        print(f"  RMSE: {rmse:.3f} cycles")
        print(f"  MAE:  {mae:.3f} cycles")
        print(f"  R²:   {r2:.3f}")
    
    results_df = pd.DataFrame(results)
    print("\n" + "-"*80)
    print("MODEL COMPARISON SUMMARY")
    print("-"*80)
    print(results_df.to_string(index=False))
    
    return results_df


def plot_predictions_vs_actual(models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Create scatter plots of predicted vs actual RUL for all models.
    """
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        
        ax.scatter(y_test, y_pred, alpha=0.6, s=30)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual RUL (cycles)', fontsize=11)
        ax.set_ylabel('Predicted RUL (cycles)', fontsize=11)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/05_predictions_vs_actual.png', dpi=300)
    print(f"\nPrediction comparison plot saved to '{OUTPUT_DIR}/05_predictions_vs_actual.png'")
    plt.close()


def plot_residuals(models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Create residual plots for all models to diagnose model fit.
    """
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        ax.scatter(y_pred, residuals, alpha=0.6, s=30)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        
        ax.set_xlabel('Predicted RUL (cycles)', fontsize=11)
        ax.set_ylabel('Residual (cycles)', fontsize=11)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/05_residual_plots.png', dpi=300)
    print(f"Residual plots saved to '{OUTPUT_DIR}/05_residual_plots.png'")
    plt.close()


# =============================================================================
# SECTION 6: FEATURE IMPORTANCE ANALYSIS
# =============================================================================

def plot_feature_importance(models: Dict[str, Any], feature_names: List[str]) -> None:
    """
    Plot feature importance for tree-based models.
    
    This helps understand which sensors/features are most important for
    predicting RUL, which could inform sensor maintenance strategies.
    """
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (name, model) in zip(axes, models.items()):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-10:]  # Top 10 features
            
            ax.barh(np.array(feature_names)[indices], importances[indices])
            ax.set_xlabel('Importance', fontsize=11)
            ax.set_title(f'{name} - Top 10 Features', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/06_feature_importance.png', dpi=300)
    print(f"Feature importance plot saved to '{OUTPUT_DIR}/06_feature_importance.png'")
    plt.close()


# =============================================================================
# SECTION 7: TEST SET PREDICTIONS
# =============================================================================

def predict_test_engines(best_model: Any, df_test: pd.DataFrame, df_test_features: pd.DataFrame,
                        rul_test_file: str) -> None:
    """
    Make RUL predictions for all test engines at their final cycle.
    
    According to assignment requirements:
    - Make exactly 100 predictions (one per test engine)
    - Use the final cycle of each test engine
    - Compare with ground truth RUL values
    
    Parameters:
    -----------
    best_model : Any
        Best trained model
    df_test : pd.DataFrame
        Original test data
    df_test_features : pd.DataFrame
        Preprocessed test features
    rul_test_file : str
        Path to RUL ground truth file
    """
    print("\n" + "="*80)
    print("SECTION 7: TEST SET PREDICTIONS")
    print("="*80)
    
    # Get the last cycle for each test engine
    last_cycle_idx = df_test.groupby('engine')['cycle'].idxmax()
    X_test_final = df_test_features.loc[last_cycle_idx].values
    engines = df_test.loc[last_cycle_idx, 'engine'].values
    
    # Make predictions
    rul_predictions = best_model.predict(X_test_final)
    rul_predictions = np.maximum(rul_predictions, 0)  # No negative RUL
    rul_predictions = np.minimum(rul_predictions, RUL_CAP)  # Cap at 125
    
    # Load ground truth
    rul_truth = pd.read_csv(rul_test_file, header=None, names=['RUL'])[['RUL']].values.flatten()
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Engine': engines,
        'Predicted_RUL': rul_predictions,
        'Actual_RUL': rul_truth
    })
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(rul_truth, rul_predictions))
    mae = mean_absolute_error(rul_truth, rul_predictions)
    r2 = r2_score(rul_truth, rul_predictions)
    
    print(f"\n7.1 Test Set Results (100 engines)")
    print(f"RMSE: {rmse:.3f} cycles")
    print(f"MAE:  {mae:.3f} cycles")
    print(f"R²:   {r2:.3f}")
    
    print(f"\n7.2 Sample Predictions (first 10 engines):")
    print(results_df.head(10).to_string(index=False))
    
    # Save results
    results_df.to_csv(f'{OUTPUT_DIR}/07_test_predictions.csv', index=False)
    print(f"\nTest predictions saved to '{OUTPUT_DIR}/07_test_predictions.csv'")
    
    # Plot test predictions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    axes[0].scatter(results_df['Actual_RUL'], results_df['Predicted_RUL'], alpha=0.6, s=40)
    min_val = min(results_df['Actual_RUL'].min(), results_df['Predicted_RUL'].min())
    max_val = max(results_df['Actual_RUL'].max(), results_df['Predicted_RUL'].max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual RUL (cycles)', fontsize=11)
    axes[0].set_ylabel('Predicted RUL (cycles)', fontsize=11)
    axes[0].set_title('Test Set: Predicted vs Actual RUL', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Error distribution
    errors = results_df['Actual_RUL'] - results_df['Predicted_RUL']
    axes[1].hist(errors, bins=20, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
    axes[1].set_xlabel('Prediction Error (cycles)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title(f'Error Distribution (RMSE={rmse:.2f})', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/07_test_predictions_analysis.png', dpi=300)
    print(f"Test analysis plot saved to '{OUTPUT_DIR}/07_test_predictions_analysis.png'")
    plt.close()
    
    return results_df


# =============================================================================
# MAIN EXECUTION WORKFLOW
# =============================================================================

def main():
    """
    Execute the complete RUL prediction pipeline.
    """
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + " PART 2: PREDICTIVE MAINTENANCE - RUL PREDICTION ".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\n Loading data...")
    df_train, df_test = load_data()
    print(f"✓ Loaded training set: {df_train.shape}")
    print(f"✓ Loaded test set: {df_test.shape}")
    
    # =========================================================================
    # STEP 2: Exploratory Data Analysis (Section 1)
    # =========================================================================
    explore_data(df_train, df_test)
    
    # =========================================================================
    # STEP 3: Data Preprocessing (Section 2)
    # =========================================================================
    df_train_features, df_test_features, scaler = preprocess_data(df_train, df_test)
    
    # =========================================================================
    # STEP 4: RUL Labeling & Data Splitting (Section 3)
    # =========================================================================
    X_train, y_train, engine_ids_train = prepare_training_data(df_train, df_train_features)
    X_train, y_train, X_val, y_val, X_test_internal, y_test_internal = split_data_by_engine(
        X_train, y_train, engine_ids_train, train_ratio=0.6, val_ratio=0.2
    )
    
    # =========================================================================
    # STEP 5: Model Training & Hyperparameter Tuning (Section 4)
    # =========================================================================
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val)

    models = {
        'Random Forest': rf_model
    }
    
    # =========================================================================
    # STEP 6: Model Evaluation (Section 5)
    # =========================================================================
    results_df = evaluate_models(models, X_test_internal, y_test_internal)
    plot_predictions_vs_actual(models, X_test_internal, y_test_internal)
    plot_residuals(models, X_test_internal, y_test_internal)
    
    # Best model is Random Forest (usually best for this type of problem)
    best_model = rf_model
    
    # =========================================================================
    # STEP 7: Feature Importance (Section 6)
    # =========================================================================
    feature_cols = [col for col in df_train_features.columns]
    plot_feature_importance(models, feature_cols)
    
    # =========================================================================
    # STEP 8: Final Test Set Predictions (Section 7)
    # =========================================================================
    # Use repository-relative paths for data and outputs
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    rul_test_file = os.path.join(base_dir, 'CMAPSSData', 'RUL_FD001.txt')
    test_results = predict_test_engines(best_model, df_test, df_test_features, rul_test_file)
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY & CONCLUSIONS")
    print("="*80)
    
    print("\n Best Model: Random Forest Regressor")
    print(f"   - Hyperparameters: {rf_model.get_params()}")
    
    # Get test performance
    y_test_pred = best_model.predict(df_test_features.values)
    rul_truth = pd.read_csv(rul_test_file, header=None, names=['RUL'])[['RUL']].values.flatten()
    
    # Get final cycle predictions only
    last_cycle_idx = df_test.groupby('engine')['cycle'].idxmax()
    X_final = df_test_features.loc[last_cycle_idx].values
    final_predictions = best_model.predict(X_final)
    
    final_rmse = np.sqrt(mean_squared_error(rul_truth, final_predictions))
    final_mae = mean_absolute_error(rul_truth, final_predictions)
    
    print(f"\n Final Test Set Performance (100 engines):")
    print(f"   - RMSE: {final_rmse:.3f} cycles")
    print(f"   - MAE:  {final_mae:.3f} cycles")
    print(f"   - Status: {'PASSED' if final_rmse <= 20 else '✗ NEEDS IMPROVEMENT'} (target: RMSE ≤ 20)")
    print(f"✓ Results saved to '{OUTPUT_DIR}/' directory")
    
    return best_model


if __name__ == "__main__":
    best_model = main()
