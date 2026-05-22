import os
from typing import Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from reliability.Fitters import Fit_Everything
from reliability.Distributions import (
    Lognormal_Distribution, Gamma_Distribution, Loglogistic_Distribution, 
    Weibull_Distribution, Normal_Distribution, Exponential_Distribution, 
    Gumbel_Distribution
)
from scipy.integrate import cumulative_trapezoid

# Import data
from import_data import load_data

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================
OUTPUT_DIR = "Output"

DIST_MAPPING = {
    'Gamma_3P': (Gamma_Distribution, ['alpha', 'beta', 'gamma']),
    'Lognormal_3P': (Lognormal_Distribution, ['mu', 'sigma', 'gamma']),
    'Lognormal_2P': (Lognormal_Distribution, ['mu', 'sigma']),
    'Loglogistic_3P': (Loglogistic_Distribution, ['alpha', 'beta', 'gamma']),
    'Weibull_3P': (Weibull_Distribution, ['alpha', 'beta', 'gamma']),
    'Loglogistic_2P': (Loglogistic_Distribution, ['alpha', 'beta']),
    'Gamma_2P': (Gamma_Distribution, ['alpha', 'beta']),
    'Normal_2P': (Normal_Distribution, ['mu', 'sigma']),
    'Weibull_2P': (Weibull_Distribution, ['alpha', 'beta']),
    'Exponential_2P': (Exponential_Distribution, ['Lambda', 'gamma']),
    'Gumbel_2P': (Gumbel_Distribution, ['mu', 'sigma']),
    'Exponential_1P': (Exponential_Distribution, ['Lambda'])
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def reconstruct_model(results: Any, dist_name: str) -> Optional[Any]:
    """
    Fetches parameters from Fit_Everything results and returns the instantiated distribution.
    """
    if dist_name not in DIST_MAPPING:
        print(f"[Warning] '{dist_name}' requires complex reconstruction or is not mapped. Skipping.")
        return None
        
    dist_class, param_names = DIST_MAPPING[dist_name]
    params = {
        p: getattr(results, f"{dist_name}_{'lambda' if p == 'Lambda' else p}") 
        for p in param_names
    }
    
    return dist_class(**params)

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
def plot_failure_times(failure_times: np.ndarray) -> None:
    print(f"Extracted {len(failure_times)} engine failure lifetimes.")
    print(f"Min: {min(failure_times)}, Q1: {np.quantile(failure_times, 0.25, method='midpoint')}, "
          f"Median: {np.quantile(failure_times, 0.5, method='midpoint')}, "
          f"Q3: {np.quantile(failure_times, 0.75, method='midpoint')}, Max: {max(failure_times)}")
    print(f"Mean: {np.mean(failure_times):.2f}, Std: {np.std(failure_times):.2f}")
    
    plt.hist(failure_times, bins=20, edgecolor='black')
    plt.xlabel('Failure Time [cycles]')
    plt.ylabel('Frequency of Engine Failure times [-]')
    plt.title('Distribution of Engine Failure Times')
    plt.savefig(f'{OUTPUT_DIR}/failure_times_histogram.png')
    plt.close()


def fit_distributions(failure_times: np.ndarray) -> Any:
    # Fitting distributions available in the reliability package to the data. Using MLE and best optimizer.
    results = Fit_Everything(
        failures=failure_times, optimizer='best', sort_by='AICc', 
        show_probability_plot=False, show_histogram_plot=False, 
        show_PP_plot=False, show_best_distribution_probability_plot=False, 
        print_results=False
    )
    return results


def distribution_analysis(failure_times: np.ndarray, results: Any, top_n: int = 5) -> None:
    """
    Automatically extracts the top N distributions from the results table, 
    dynamically reconstructs them, and plots both their PDFs and CDFs 
    against the empirical data side-by-side.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Prepare Empirical Data
    ax1.hist(failure_times, bins=15, density=True, alpha=0.3, color='grey', 
             edgecolor='black', label='Empirical Data')
    
    sorted_failures = np.sort(failure_times)
    y_ecdf = np.arange(1, len(sorted_failures) + 1) / len(sorted_failures)
    ax2.step(sorted_failures, y_ecdf, where='post', color='grey', 
             linewidth=3, label='Empirical CDF')
    
    # Dynamically scale time grid (prevent negative cycles)
    min_time = max(0, min(failure_times) - 20)
    t_grid = np.linspace(min_time, max(failure_times) + 20, 1000)
    
    # 2. Reconstruct and plot the top contenders
    top_dist_names = results.results['Distribution'].head(top_n).tolist()
    
    for name in top_dist_names:
        model = reconstruct_model(results, name)
        if not model:
            continue
            
        pdf_values = model.PDF(xvals=t_grid, show_plot=False)
        ax1.plot(t_grid, pdf_values, label=name, linewidth=2)
        
        cdf_values = model.CDF(xvals=t_grid, show_plot=False)
        ax2.plot(t_grid, cdf_values, label=name, linewidth=2)

    # 3. Formatting
    for ax, title, ylabel in zip([ax1, ax2], ['PDF', 'CDF'], ['Probability Density $f(t)$', 'Cumulative Probability $F(t)$']):
        ax.set_xlabel('Time to Failure $t_f$ [flight cycles]', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{title} Comparison (Top {top_n} Contenders)', fontsize=13, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        
    ax1.legend(fontsize=10, loc='upper left')
    ax2.legend(fontsize=10, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/top_contenders_pdf_cdf_comparison.png', dpi=300)
    print(f"Visual comparison plot successfully saved to '{OUTPUT_DIR}/top_contenders_pdf_cdf_comparison.png'.")
    plt.close()


def plot_hazard_functions(failure_times: np.ndarray, results: Any, top_n: int = 5) -> None:
    """
    Automatically extracts the top N distributions from the results table, 
    dynamically reconstructs them, and plots their Hazard Functions, h(t).
    """
    plt.figure(figsize=(10, 6))
    t_grid = np.linspace(0, max(failure_times) + 50, 1000)
    top_dist_names = results.results['Distribution'].head(top_n).tolist()
    
    for name in top_dist_names:
        model = reconstruct_model(results, name)
        if not model:
            continue
            
        hf_values = model.HF(xvals=t_grid, show_plot=False)
        plt.plot(t_grid, hf_values, label=name, linewidth=2.5)

    plt.xlabel('Time to Failure $t$ [flight cycles]', fontsize=12)
    plt.ylabel('Hazard Rate $h(t)$', fontsize=12)
    plt.title(f'Hazard Function Analysis (Top {top_n} Contenders)', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10, loc='upper left')
    plt.ylim(bottom=0) 
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/hazard_functions_comparison.png', dpi=300)
    print(f"Hazard function plot successfully saved to '{OUTPUT_DIR}/hazard_functions_comparison.png'.")
    plt.close()


def compare_optimal_maintenance(failure_times: np.ndarray, results: Any, Cp: int = 10000, Cc: int = 100000, top_n: int = 4) -> None:
    """
    Dynamically reconstructs the top N distributions, calculates their long-term 
    average maintenance cost g(t), and compares their optimal replacement times.
    """
    plt.figure(figsize=(12, 7))
    
    # Scale grid dynamically based on empirical maximum life
    max_time = max(failure_times) + 50
    t_grid = np.linspace(0, max_time, 4000)
    
    top_dist_names = results.results['Distribution'].head(top_n).tolist()
    
    print("\n===================================================================")
    print("           COMPARATIVE MAINTENANCE OPTIMIZATION (g(t))             ")
    print("===================================================================")
    print(f"{'Model Name':<20} | {'Optimal Time (t*)':<20} | {'Minimum Cost (€/cycle)'}")
    print("-" * 67)
    
    max_cost_opt = 0 
    
    for name in top_dist_names:
        model = reconstruct_model(results, name)
        if not model:
            continue
            
        R_t = model.SF(xvals=t_grid, show_plot=False)
        F_t = model.CDF(xvals=t_grid, show_plot=False)
        expected_length = cumulative_trapezoid(R_t, t_grid, initial=0)
        
        # Slice to avoid division by zero at t=0
        t_eval = t_grid[1:]
        R_eval = R_t[1:]
        F_eval = F_t[1:]
        len_eval = expected_length[1:]
        
        g_t = (Cp * R_eval + Cc * F_eval) / len_eval
        
        min_idx = np.argmin(g_t)
        t_opt = t_eval[min_idx]
        cost_opt = g_t[min_idx]
        
        if cost_opt > max_cost_opt:
            max_cost_opt = cost_opt
            
        line, = plt.plot(t_eval, g_t, label=f'{name}', linewidth=2)
        plt.plot(t_opt, cost_opt, marker='o', markersize=6, color=line.get_color())
        
        print(f"{name:<20} | {t_opt:>12.2f} cycles | € {cost_opt:>17.2f}")

    print("===================================================================\n")

    plt.xlabel('Preventive Replacement Time $t$ [flight cycles]', fontsize=12)
    plt.ylabel('Long-Term Average Cost $g(t)$ [€ / cycle]', fontsize=12)
    plt.title(f'Cost Optimization Comparison: Top {top_n} Contenders', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10, loc='upper right')
    
    plt.ylim(0, max_cost_opt * 3)
    # Dynamically bind the x-axis to the logical range of the dataset instead of hardcoded 300
    plt.xlim(0, max(failure_times)) 
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/comparative_maintenance_cost.png', dpi=300)
    print(f"Comparative cost plot successfully saved to '{OUTPUT_DIR}/comparative_maintenance_cost.png'.")
    plt.close()

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    # Ensure output directory exists before attempting to save files
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load the data
    df_train, _ = load_data()
    
    # 2. Extract failure times (max cycle per engine)
    failure_times = df_train.groupby('engine')['cycle'].max().values
    
    # 3. Fit distributions using the reliability package
    results = fit_distributions(failure_times)

    # 4. Analysis & Visualizations
    top_n = 5
    # plot_failure_times(failure_times)
    distribution_analysis(failure_times, results, top_n)
    plot_hazard_functions(failure_times, results, top_n)

    # Note: Updated this function call to pass failure_times for the dynamic grid
    compare_optimal_maintenance(failure_times, results, Cp=10000, Cc=100000, top_n=top_n)


if __name__ == "__main__":
    main()