import matplotlib.pyplot as plt
import numpy as np
from reliability.Fitters import Fit_Everything
from reliability.Distributions import (Lognormal_Distribution, Gamma_Distribution, Loglogistic_Distribution, Weibull_Distribution, Normal_Distribution, Exponential_Distribution, Gumbel_Distribution)
# Import data
from import_data import load_data

def plot_failure_times(failure_times):
    # Basic statistics and histogram of failure times
    print(f"Extracted {len(failure_times)} engine failure lifetimes.")

    print(f"Min: {min(failure_times)}, Q1: {np.quantile(failure_times, 0.25, method='midpoint')}, Median: {np.quantile(failure_times, 0.5, method='midpoint')}, Q3: {np.quantile(failure_times, 0.75, method='midpoint')}, Max: {max(failure_times)}")
    print(f"Mean: {np.mean(failure_times)}, Std: {np.std(failure_times)}")
    plt.hist(failure_times, bins=20, edgecolor='black')
    plt.xlabel('Failure Time [cycles]')
    plt.ylabel('Frequency of Engine Failure times [-]')
    plt.title('Distribution of Engine Failure Times')
    plt.savefig('Output/failure_times_histogram.png')


def fit_distributions(failure_times):
    # Fitting distributions available in the reliability package to the data. Using MLE and best optimizer.
    results = Fit_Everything(failures=failure_times, optimizer = 'best', sort_by='AICc', show_probability_plot = False, show_histogram_plot = False, show_PP_plot =False, show_best_distribution_probability_plot = False, print_results = True)
    # hist_plot = results.histogram_plot
    # hist_plot.savefig('Output/failure_times_fitted_histogram.png')

    return results


def distribution_analysis(failure_times, results, top_n=5):
    """
    Automatically extracts the top N distributions from the results table, 
    dynamically reconstructs them, and plots both their PDFs and CDFs 
    against the empirical data side-by-side using solid lines.
    """
    # Create a figure with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Prepare Empirical Data
    # For PDF (Subplot 1)
    ax1.hist(failure_times, bins=15, density=True, alpha=0.3, color='grey', 
             edgecolor='black', label='Empirical Data')
    
    # For CDF (Subplot 2) - Calculate Empirical CDF (ECDF)
    sorted_failures = np.sort(failure_times)
    y_ecdf = np.arange(1, len(sorted_failures) + 1) / len(sorted_failures)
    ax2.step(sorted_failures, y_ecdf, where='post', color='grey', 
             linewidth=3, label='Empirical CDF')
    
    t_grid = np.linspace(min(failure_times) - 20, max(failure_times) + 20, 1000)
    
    # 2. Comprehensive map of standard string names to actual classes and arguments
    # This ensures ANY standard distribution that makes the top N can be plotted
    dist_mapping = {
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
    
    # 3. Get TOP N distribution names from the sorted table
    top_dist_names = results.results['Distribution'].head(top_n).tolist()
    
    # 4. Reconstruct and plot the top contenders
    for name in top_dist_names:
        if name in dist_mapping:
            dist_class, param_names = dist_mapping[name]
            
            # Fetch parameters dynamically 
            params = {p: getattr(results, f"{name}_{p}") for p in param_names}
            model = dist_class(**params)
            
            # Plot PDF on ax1 (solid lines, relying on matplotlib's default color cycle)
            pdf_values = model.PDF(xvals=t_grid, show_plot=False)
            ax1.plot(t_grid, pdf_values, label=name, linewidth=2)
            
            # Plot CDF on ax2 
            cdf_values = model.CDF(xvals=t_grid, show_plot=False)
            ax2.plot(t_grid, cdf_values, label=name, linewidth=2)
        else:
            print(f"[Warning] '{name}' is in the top {top_n} but requires complex reconstruction. Skipping.")

    # 5. Formatting Subplot 1 (PDF)
    ax1.set_xlabel('Time to Failure $t_f$ [flight cycles]', fontsize=12)
    ax1.set_ylabel('Probability Density $f(t)$', fontsize=12)
    ax1.set_title(f'PDF Comparison (Top {top_n} Contenders)', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=10, loc='upper left')
    
    # 6. Formatting Subplot 2 (CDF)
    ax2.set_xlabel('Time to Failure $t_f$ [flight cycles]', fontsize=12)
    ax2.set_ylabel('Cumulative Probability $F(t)$', fontsize=12)
    ax2.set_title(f'CDF Comparison (Top {top_n} Contenders)', fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=10, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('Output/top_contenders_pdf_cdf_comparison.png', dpi=300)
    print(f"Visual comparison plot successfully saved to 'Output/top_contenders_pdf_cdf_comparison.png'.")



def main():
    # 1. Load the data
    df_train, _ = load_data()
    
    # 2. Extract failure times (max cycle per engine) for Part 1
    # Group by engine and find the maximum cycle number where it failed
    failure_times = df_train.groupby('engine')['cycle'].max().values
    
    # 3. Getting some basic statistics and visualizations of the failure times
    # plot_failure_times(failure_times)
    
    # 4. Fit distributions using the reliability package
    results = fit_distributions(failure_times)

    # 5. Analysis of best fitting distribution
    distribution_analysis(failure_times, results, top_n=5)
    
    # 6. Implement your cost optimization function g(t) here
    # ...

if __name__ == "__main__":
    main()