# AE4465
This is an assignment for the AE4465 Maintenance Modeling and Analysis course of the TU Delft Aerospace Engineering Master in the track Operations and Environment.

## Project Structure

**Data Handling:**
`import_data.py`: Utility module responsible for loading, parsing, and structuring the raw CMAPSS dataset. It is imported by the other scripts to ensure data consistency across all models.

**Part 1: Preventive Maintenance Modeling:**
`part1_preventive.py`: Focuses on foundational reliability analysis and preventive maintenance strategies

**Part 2: Predictive Modeling (Remaining Useful Life):**
Predicts the Remaining Useful Life (RUL) of turbofan engines dynamically using time-series sensor data.
`part2_predictive.py`: Classical machine learning baseline Random Forest model 
`part2_predictive_NN.py`: Deep Learning using `PyTorch`. Implements a Long Short-Term Memory (LSTM) network designed to handle complex sequential time-series data.

**Part 3: Uncertainty Quantification (UQ):**
`part3_uq.py`: Focuses on Uncertainty Quantification. Analyzes the variance, probability distributions, and confidence intervals of the predictive models to aid in robust, risk-averse maintenance decision-making.
