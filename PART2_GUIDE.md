# Part 2: Predictive Maintenance - RUL Prediction Guide

## Overview

This guide explains the `part2_predictive.py` script, which implements a machine learning-based Remaining Useful Life (RUL) prediction system for aircraft engines using the C-MAPSS dataset.

## File Structure

```
src/
├── import_data.py              # Data loading utilities
├── part1_preventive.py         # Part 1: Preventive maintenance (reliability analysis)
├── part2_predictive.py         # Part 2: Predictive maintenance (RUL prediction) ← YOU ARE HERE
└── __pycache__/
```

## Workflow Overview

The script follows a structured 8-step workflow:

```
1. Data Loading & Exploration
        ↓
2. Data Preprocessing (Standardization)
        ↓
3. RUL Calculation & Data Splitting (by engine)
        ↓
4. Model Training & Hyperparameter Tuning
        ├── Random Forest Regressor
        ├── Gradient Boosting Regressor
        └── Ridge Regression (baseline)
        ↓
5. Model Evaluation (Train/Val/Test)
        ↓
6. Feature Importance Analysis
        ↓
7. Test Set Predictions (100 engines)
        ↓
8. Results Visualization & Summary
```

---

## Section-by-Section Breakdown

### SECTION 1: Data Exploration & Analysis

**Purpose:** Understand the dataset and identify potential issues.

**Key Analyses:**
- Dataset shape: 20,631 training samples, 13,096 test samples
- Number of engines: 100 training, 100 test
- Missing values check
- Feature statistics (mean, std, min, max)
- Feature variance analysis (identify constant features)
- Correlation matrix (identify redundant features)

**Output:**
- Console output with statistics
- `01_feature_correlations.png` - Heatmap of feature correlations

**Key Insight:** 
- The dataset has no missing values
- All 21 sensor/operational features are used (excluding engine ID and cycle number)
- Features with high correlation (>0.95) may be redundant, but we keep them for model robustness

---

### SECTION 2: Data Preprocessing

**Purpose:** Prepare features for machine learning models.

**Steps:**
1. **Feature Selection:** Exclude 'engine' and 'cycle' columns
   - WHY? These would leak information about RUL
   - In practice, we only have sensor measurements, not engine ID
   
2. **Low-Variance Feature Removal:** Remove features with variance < 0.01
   - Constant features provide no predictive information
   
3. **Feature Standardization:** Scale all features to zero mean, unit variance
   - WHY? ML algorithms perform better with normalized features
   - Formula: `z = (x - mean) / std`

**Code Example:**
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Output:**
- Standardized feature matrices ready for modeling

---

### SECTION 3: RUL Calculation & Data Splitting

**Purpose:** Create target labels and ensure proper data partitioning.

#### 3.1: RUL Calculation

**What is RUL?**
- Remaining Useful Life = cycles remaining until failure
- Calculated as: `RUL = max_cycle_for_engine - current_cycle`
- Capped at 125 cycles

**Why cap at 125?**
1. Healthy engines (high RUL) are difficult to predict accurately
2. A prediction close to 125 indicates "engine is healthy"
3. Helps model focus on degradation patterns
4. Matches real-world operational constraints

**Example:**
```
Engine 1 fails at cycle 150:
  - At cycle 50: RUL = 150 - 50 = 100 cycles
  - At cycle 140: RUL = 150 - 140 = 10 cycles
  - At cycle 200 (if healthy): RUL → capped at 125
```

#### 3.2: Data Splitting (By Engine)

**Critical Requirement:** Prevent data leakage

**Problem:** Engines have unique degradation patterns
- If engine #5 appears in both training and test sets, the model can memorize its signature
- In practice, we wouldn't have the same engine in both sets

**Solution:** Split by engine (not random samples)
```
Split: 60% training engines, 20% validation engines, 20% test engines
Result:
  - Training: ~60 engines (12,000+ samples)
  - Validation: ~20 engines (4,000+ samples)
  - Test: ~20 engines (4,000+ samples)
```

**Code Visualization:**
```python
# Get unique engines
unique_engines = [1, 2, 3, ..., 100]

# Shuffle and split by engine (NOT random rows!)
train_engines = [1, 3, 5, 7, ...]  # 60 engines
val_engines = [2, 4, 6, 8, ...]    # 20 engines
test_engines = [10, 12, 14, ...]   # 20 engines

# Select all samples from training engines
X_train = X[engine_id in train_engines]
```

---

### SECTION 4: Model Training & Hyperparameter Tuning

**Purpose:** Train and optimize three different ML models.

#### 4A: Random Forest Regressor

**Why Random Forest?**
- ✓ Non-parametric (no assumptions about data distribution)
- ✓ Handles non-linear relationships
- ✓ Robust to outliers
- ✓ Provides feature importance
- ✓ Fast training and prediction
- ✓ Excellent for RUL prediction tasks

**Hyperparameters Tuned:**
```python
param_grid = {
    'n_estimators': [50, 100, 200],           # Number of trees
    'max_depth': [10, 20, 30, None],          # Tree depth (controls complexity)
    'min_samples_split': [2, 5, 10],          # Regularization
    'min_samples_leaf': [1, 2, 4]             # Regularization
}
```

**Hyperparameter Explanation:**
- `n_estimators`: More trees = better fit, but slower and risk overfitting
- `max_depth`: Deeper trees = more complex patterns, but risk overfitting
- `min_samples_split`: Require more samples to split = simpler trees
- `min_samples_leaf`: Require more samples in leaf = smoother predictions

**Tuning Method:** GridSearchCV with 3-fold cross-validation on training set

#### 4B: Gradient Boosting Regressor

**Why Gradient Boosting?**
- ✓ Sequential improvement of weak learners
- ✓ Often achieves state-of-the-art performance
- ✓ Excellent at capturing complex patterns
- ✓ Provides feature importance

**Hyperparameters Tuned:**
```python
param_grid = {
    'n_estimators': [100, 200, 300],          # Boosting stages
    'learning_rate': [0.01, 0.05, 0.1],       # Step size
    'max_depth': [3, 5, 7],                   # Individual tree depth
    'min_samples_split': [2, 5],              # Regularization
    'min_samples_leaf': [1, 2]                # Regularization
}
```

#### 4C: Ridge Regression (Baseline)

**Why Ridge Regression?**
- ✓ Simple, fast, interpretable
- ✓ Good baseline for comparison
- ✓ L2 regularization prevents overfitting
- ✓ Linear model for reference

---

### SECTION 5: Model Evaluation

**Purpose:** Assess model performance on held-out test set.

**Evaluation Metrics:**

1. **RMSE (Root Mean Squared Error)**
   ```
   RMSE = sqrt(mean((actual - predicted)²))
   - Units: cycles
   - Lower is better
   - TARGET: ≤ 20 cycles for full credit
   - Penalizes large errors more heavily
   ```

2. **MAE (Mean Absolute Error)**
   ```
   MAE = mean(|actual - predicted|)
   - Units: cycles
   - Lower is better
   - Average prediction error in cycles
   ```

3. **R² Score**
   ```
   R² = 1 - (SS_res / SS_tot)
   - Range: -∞ to 1.0
   - 1.0 = perfect prediction
   - 0.0 = model performs as well as mean baseline
   - Negative = worse than baseline
   ```

**Visualizations:**
- `05_predictions_vs_actual.png` - Scatter plots (predicted vs actual)
- `05_residual_plots.png` - Residual analysis

**Residual Analysis Interpretation:**
- Should be randomly scattered around zero
- No patterns = good fit
- U-shaped pattern = underfitting
- Systematic bias = model has room for improvement

---

### SECTION 6: Feature Importance Analysis

**Purpose:** Understand which sensors are most important for RUL prediction.

**Output:**
- `06_feature_importance.png` - Top 10 features for each tree-based model

**Interpretation Example:**
```
Top sensors might include:
1. T50 (Temperature at LPT outlet) - Strong indicator of degradation
2. NRc (Corrected core speed) - Engine load indicator
3. P30 (HPC outlet pressure) - Compression indicator
4. W32 (LPT coolant bleed) - Cooling system health
```

**Business Value:**
- Focus maintenance resources on monitoring important sensors
- Discard sensors with low importance (cost reduction)
- Understand physical degradation mechanisms

---

### SECTION 7: Test Set Predictions

**Purpose:** Make final RUL predictions for 100 test engines.

**Procedure:**
1. Extract the LAST CYCLE of each test engine
2. Use standardized features from that cycle
3. Predict RUL using best trained model
4. Compare with ground truth RUL values

**Output:**
- `07_test_predictions.csv` - Predictions and ground truth for all 100 engines
- `07_test_predictions_analysis.png` - Scatter plot and error distribution

**Example Output:**
```
Engine | Predicted_RUL | Actual_RUL | Error
-------|---------------|------------|------
    1  |      45.2     |    42      | 3.2
    2  |     125.0     |   127      | -2.0
    3  |      15.8     |    18      | -2.2
   ...
```

---

## How to Run the Script

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Execution
```bash
cd src/
python part2_predictive.py
```

### Console Output
The script prints:
- Section headers and progress
- Dataset statistics
- Model hyperparameters
- Validation/test performance metrics
- Summary statistics

### Generated Files
Saved to `Output/` directory:
- `01_feature_correlations.png` - Feature relationships
- `05_predictions_vs_actual.png` - Model accuracy visualization
- `05_residual_plots.png` - Residual diagnostics
- `06_feature_importance.png` - Sensor importance ranking
- `07_test_predictions.csv` - Final predictions (100 engines)
- `07_test_predictions_analysis.png` - Test set analysis

---

## Key Design Decisions

### 1. Why exclude 'engine' and 'cycle'?
The instructions specifically state:
> "For your final RUL prediction model, do NOT use the engine number or cycle number as input"

**Reason:** 
- In real deployment, we only have sensor measurements
- These columns would leak lifetime information
- Cycle number = RUL information → invalid model

### 2. Why cap RUL at 125?
- Healthy engines are difficult to predict
- Represents "essentially healthy" state
- Improves model focus on degradation patterns

### 3. Why split by engine (not random)?
- Prevents data leakage
- Reflects real-world validation scenario
- Ensures true model generalization

### 4. Why three models?
- **Random Forest:** Best practical performance
- **Gradient Boosting:** Potentially better but slower
- **Ridge:** Fast baseline for comparison

### 5. Why GridSearchCV?
- Systematic hyperparameter optimization
- Cross-validation on training data
- Prevents overfitting to validation set

---

## Interpretation Guide

### Perfect Performance
```
RMSE: 0 cycles  ← Predictions perfectly match reality
R²: 1.0         ← 100% variance explained
```

### Good Performance
```
RMSE: < 10 cycles
R²: 0.8-0.95
→ Model captures degradation patterns well
```

### Target Performance
```
RMSE: ≤ 20 cycles (assignment requirement)
R²: 0.6-0.8
→ Practically useful predictions
```

### Poor Performance
```
RMSE: > 50 cycles
R²: < 0.3
→ Model struggles; consider different approach
```

---

## Advanced Topics

### Feature Engineering
Current features are direct sensor measurements. Could add:
- Polynomial features (x²)
- Interaction terms (temp × speed)
- Rolling statistics (moving averages)
- Rate of change (derivative approximation)

### Class Imbalance
RUL distribution is heavily skewed:
- Many "healthy" engines (RUL ≈ 125)
- Few "degraded" engines (RUL < 20)

Solutions:
- Weight samples by RUL range
- Use quantile loss instead of MSE
- Stratified sampling

### Overfitting Prevention
Beyond hyperparameter tuning:
- Early stopping (monitor validation loss)
- Dropout for tree models (random subspace methods)
- Ensemble different architectures

### Explainability
LIME (Local Interpretable Model-agnostic Explanations):
- Explain individual predictions
- "Why did model predict RUL = 45 for engine 5?"
- Identify critical features for each engine

---

## Troubleshooting

### Issue: RMSE > 20
**Solutions:**
1. Check data preprocessing (are features scaled correctly?)
2. Try different hyperparameter ranges
3. Feature engineering (add new features)
4. Ensemble multiple models
5. Collect more training data

### Issue: Model overfitting (train RMSE << test RMSE)
**Solutions:**
1. Increase regularization (max_depth, min_samples_split)
2. Reduce learning_rate for Gradient Boosting
3. Reduce n_estimators
4. Add more training data

### Issue: Model underfitting (both train and test RMSE high)
**Solutions:**
1. Decrease regularization
2. Increase model complexity (deeper trees)
3. Increase n_estimators
4. Add feature engineering

---

## References

### Mathematical Formulas

**Remaining Useful Life:**
```
RUL(t) = min(max_cycle - t, 125)
where t = current cycle
```

**Standardization:**
```
z = (x - μ) / σ
where μ = mean, σ = standard deviation
```

**RMSE:**
```
RMSE = sqrt((1/n) * Σ(ŷᵢ - yᵢ)²)
where ŷᵢ = prediction, yᵢ = actual
```

### Recommended Reading
- Scikit-learn documentation: https://scikit-learn.org
- Random Forest explainer: https://towardsdatascience.com/random-forest-in-python
- Gradient Boosting: https://kaggle.com/learn/intro-to-machine-learning

---

## Questions?

See Part 2 of Assignment2.pdf for full requirements and grading criteria.

**Assessment Criteria (75 points total):**
1. Data Description (10 points): ✓ Covered in Section 1
2. Model Description & Training (30 points): ✓ Covered in Sections 2-4
3. Hyperparameter Tuning (10 points): ✓ Covered in Section 4
4. Results & RMSE ≤ 20 (25 points): ✓ Covered in Section 7

This script addresses all requirements with clear explanations and visualizations.
