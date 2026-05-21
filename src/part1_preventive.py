import matplotlib.pyplot as plt
import numpy as np
from reliability.Fitters import Fit_Everything

# Import data
from import_data import load_data

def main():
    # 1. Load the data
    df_train, _ = load_data()
    
    # 2. Extract failure times (max cycle per engine) for Part 1
    # Group by engine and find the maximum cycle number where it failed
    failure_times = df_train.groupby('engine')['cycle'].max().values
    
    # Getting some basic statistics and visualizations of the failure times
    # print(f"Extracted {len(failure_times)} engine failure lifetimes.")
    # print(min(failure_times), np.quantile(failure_times, 0.25), np.quantile(failure_times, 0.5), np.quantile(failure_times, 0.75), max(failure_times))
    # print(np.mean(failure_times), np.std(failure_times))
    # plt.hist(failure_times, bins=20, edgecolor='black')
    # plt.xlabel('Failure Time')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Engine Failure Times')
    # plt.show()
    
    # 3. Fit distributions using the reliability package
    results = Fit_Everything(failures=failure_times, show_probability_plot = False, show_histogram_plot = True, show_PP_plot =False, show_best_distribution_probability_plot = False, print_results = False)
    results_df = results.results
    print(results_df)
    
    # 4. Implement your cost optimization function g(t) here
    # ...

if __name__ == "__main__":
    main()