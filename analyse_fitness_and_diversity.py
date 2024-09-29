
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, ttest_rel, wilcoxon
import numpy as np
from statsmodels.stats.stattools import durbin_watson

# Set manual input for enemies to evaluate (e.g., [5, 6])
enemies_to_evaluate = [5, 6, 8]

# Base directories for both algorithms
algorithm_dirs = {
    'Algorithm 1': 'EA1_line_plot_runs',
    'Algorithm 2': 'matusEA2exp'
}

def process_results_for_enemy(base_folder, enemy, algorithm):
    """
    Process the results for a given enemy for one algorithm, handling different logging formats.
    """
    enemy_folder = os.path.join(base_folder, f"EN{enemy}")
    
    if not os.path.isdir(enemy_folder):
        print(f"Enemy folder not found: {enemy_folder}")
        return None

    all_dfs = []  # Initialize an empty list to hold dataframes

    # Iterate through the run folders (e.g., run_1_ENX, run_2_ENX, etc.)
    for run_folder in os.listdir(enemy_folder):
        run_path = os.path.join(enemy_folder, run_folder)
        
        if os.path.isdir(run_path):
            result_file = os.path.join(run_path, 'results.txt')  # Each run has a 'results.txt' file
            
            if os.path.exists(result_file):
                # Read results file
                if algorithm == 'Algorithm 1':
                    # For Algorithm 1
                    df = pd.read_csv(result_file, delim_whitespace=True)
                    
                    # Drop the first duplicate generation 0 row (if it exists)
                    df = df.drop_duplicates(subset=['gen'], keep='last')
                    
                elif algorithm == 'Algorithm 2':
                    # For Algorithm 2
                    df = pd.read_csv(result_file, delim_whitespace=True, comment='#', header=None)
                    df.columns = ['gen', 'best', 'mean', 'std', 'diversity']  # Add column names
                    
                    # Convert the 'gen' column to integer since it's stored as float
                    df['gen'] = df['gen'].astype(int)
                    
                # Ensure valid numeric data
                df = df[pd.to_numeric(df['best'], errors='coerce').notnull()]

                # Convert columns to numeric
                df['best'] = pd.to_numeric(df['best'])
                df['mean'] = pd.to_numeric(df['mean'])
                df['std'] = pd.to_numeric(df['std'])
                df['diversity'] = pd.to_numeric(df['diversity'])
                df['gen'] = pd.to_numeric(df['gen'])

                # Append DataFrame to the list
                all_dfs.append(df)

    if all_dfs:
        return pd.concat(all_dfs)  # Return the concatenated dataframe
    return None

def make_subplots_across_enemies(algorithm_dfs_dict, enemies):
    """
    Create subplots with the results for all enemies, with separate plots for fitness and diversity.
    """
    n_enemies = len(enemies)  # Number of enemies provided
    fig, axes = plt.subplots(nrows=n_enemies, ncols=2, figsize=(12, 18))  # Create an n x 2 grid

    if n_enemies == 1:
        axes = [axes]  # Handle the case where there's only one enemy to plot

    # Iterate over each enemy and corresponding axes
    for i, (enemy, ax_pair) in enumerate(zip(enemies, axes)):
        ax_fitness, ax_diversity = ax_pair  # Unpack the axes for fitness and diversity

        algorithm_dfs = algorithm_dfs_dict[enemy]

        for algorithm_name, df in algorithm_dfs.items():
            if df is not None:
                # Group by generation and calculate mean and std for mean and best fitness
                grouped_stats = df.groupby('gen').agg(
                    mean_avg_fitness=('mean', 'mean'),
                    std_avg_fitness=('mean', 'std'),
                    mean_max_fitness=('best', 'mean'),
                    std_max_fitness=('best', 'std'),
                    mean_diversity=('diversity', 'mean')
                )

                generations = grouped_stats.index

                # Plot mean fitness with standard deviation shading on fitness axis
                ax_fitness.plot(generations, grouped_stats['mean_avg_fitness'], label=f"{algorithm_name}: mean fitness", linestyle='--')
                ax_fitness.fill_between(generations,
                                        grouped_stats['mean_avg_fitness'] - grouped_stats['std_avg_fitness'],
                                        grouped_stats['mean_avg_fitness'] + grouped_stats['std_avg_fitness'],
                                        alpha=0.2)

                # Plot best fitness with standard deviation shading on fitness axis
                ax_fitness.plot(generations, grouped_stats['mean_max_fitness'], label=f"{algorithm_name}: best fitness", linestyle='-')
                ax_fitness.fill_between(generations,
                                        grouped_stats['mean_max_fitness'] - grouped_stats['std_max_fitness'],
                                        grouped_stats['mean_max_fitness'] + grouped_stats['std_max_fitness'],
                                        alpha=0.2)

                # Plot diversity on diversity axis
                ax_diversity.plot(generations, grouped_stats['mean_diversity'], label=f"{algorithm_name}: diversity", linestyle=':')

        # Customize the fitness plot (left)
        ax_fitness.set_title(f'Fitness evolution for enemy {enemy}', fontsize=14)
        ax_fitness.set_xlabel('Generation', fontsize=12)
        ax_fitness.set_ylabel('Fitness', fontsize=12)
        ax_fitness.grid(True)

        # Customize the diversity plot (right)
        ax_diversity.set_title(f'Diversity evolution for enemy {enemy}', fontsize=14)
        ax_diversity.set_xlabel('Generation', fontsize=12)
        ax_diversity.set_ylabel('Diversity', fontsize=12)
        ax_diversity.grid(True)
        #ax_diversity.tick_params(axis='y', colors='red')

        # Combine legends for fitness and diversity
        fitness_lines, fitness_labels = ax_fitness.get_legend_handles_labels()
        diversity_lines, diversity_labels = ax_diversity.get_legend_handles_labels()
        ax_fitness.legend(fitness_lines, fitness_labels, loc='lower right')
        ax_diversity.legend(diversity_lines, diversity_labels, loc='upper right')

    # Adjust layout to prevent overlap between subplots
    plt.tight_layout(pad=3.0)
    plt.savefig('line_plots_3x2.png')  # Uncomment to save the figure
    plt.show()

def process_across_enemies(enemies_to_evaluate, algorithm_dirs):
    """
    Process results for all enemies and both algorithms, and generate n x 2 subplots for fitness and diversity.
    """
    algorithm_dfs_dict = {}

    for enemy in enemies_to_evaluate:
        print(f"Processing enemy {enemy}...")

        # Dictionary to hold the dataframes for both algorithms
        algorithm_dfs = {}

        # Process results for both algorithms
        for algorithm_name, base_folder in algorithm_dirs.items():
            df = process_results_for_enemy(base_folder, enemy, algorithm_name)
            algorithm_dfs[algorithm_name] = df
        
        # Store the results for this enemy
        algorithm_dfs_dict[enemy] = algorithm_dfs

    return algorithm_dfs_dict

def test_normality(data):
    """
    Test if the small dataset (n<50) is normally distributed using the Shapiro-Wilk test.
    
    Parameters:
    The dataset to test.

    Returns:
    The p-value of the Shapiro-Wilk test.
    """

    stat, p_value = shapiro(data)
    print(f"Shapiro-Wilk Test Statistic: {stat}, P-value: {p_value}")
    return p_value

def test_autocorrelation(data):
    """
    Test if the dataset has significant autocorrelation using the Durbin-Watson test.
    
    Parameters:
    The dataset to test.
    
    Returns:
    Durbin-Watson statistic.
    """

    dw_stat = durbin_watson(data)
    print(f"Durbin-Watson Statistic: {dw_stat}")
    return dw_stat

def test_equal_variance(data1, data2):
    """
    Test if two datasets have equal variances using Levene's test.
    
    Parameters:
    The two datasets to test.
    
    Returns:
    The p-value of Levene's test.
    """
    stat, p_value = levene(data1, data2)
    print(f"Levene's Test Statistic: {stat}, P-value: {p_value}")
    return p_value

def compare_datasets(data1, data2):
    """
    Compare two datasets using t-test or Wilcoxon Signed-Rank Test based on normality, autocorrelation, and equal variance.
    
    Parameters:
    The two datasets to compare
    
    Returns:
    The test used and the p-value.
    """
    # Test for normality
    p_value_normal1 = test_normality(data1)
    p_value_normal2 = test_normality(data2)
    
    # Test for autocorrelation
    dw_stat1 = test_autocorrelation(data1)
    dw_stat2 = test_autocorrelation(data2)
    
    # Test for equal variance
    p_value_equal_variance = test_equal_variance(data1, data2)
    
    # Decide on the test to use
    if p_value_normal1 > 0.05 and p_value_normal2 > 0.05 and 1.5 < dw_stat1 < 2.5 and 1.5 < dw_stat2 < 2.5:
        if p_value_equal_variance > 0.05:
            # Normal, no autocorrelation, and equal variances: use paired t-test
            stat, p_value = ttest_rel(data1, data2)
            print(f"T-test used. T-statistic: {stat}, P-value: {p_value}")
            return "T-test", p_value
        else:
            # Normal, no autocorrelation, but unequal variances: use Wilcoxon Signed-Rank Test
            stat, p_value = wilcoxon(data1, data2)
            print(f"Wilcoxon Signed-Rank Test used due to unequal variances. Test statistic: {stat}, P-value: {p_value}")
            return "Wilcoxon Signed-Rank Test (unequal variances)", p_value
    else:
        # Non-normal or autocorrelated, use Wilcoxon Signed-Rank Test
        stat, p_value = wilcoxon(data1, data2)
        print(f"Wilcoxon Signed-Rank Test used. Test statistic: {stat}, P-value: {p_value}")
        return "Wilcoxon Signed-Rank Test", p_value

def perform_stats_test(algorithm_dfs_dict, enemies):
    """
    Perform statistical tests on the mean best fitness and mean diversity for each enemy.
    """
    for enemy in enemies:
        print(f"\nProcessing statistical tests for enemy {enemy}...")

        algorithm_dfs = algorithm_dfs_dict[enemy]

        # Initialize placeholders for the mean max fitness and mean diversity of both algorithms
        grouped_stats_mean_max_fitness_algo1 = None
        grouped_stats_mean_max_fitness_algo2 = None
        grouped_stats_mean_diversity_algo1 = None
        grouped_stats_mean_diversity_algo2 = None

        # Iterate through algorithms and save the stats for both algorithms
        for algorithm_name, df in algorithm_dfs.items():
            if df is not None:
                # Group by generation and calculate mean max fitness and mean diversity
                grouped_stats = df.groupby('gen').agg(
                    mean_max_fitness=('best', 'mean'),
                    mean_diversity=('diversity', 'mean')
                )

                # Store stats for the correct algorithm
                if algorithm_name == "Algorithm 1":
                    grouped_stats_mean_max_fitness_algo1 = grouped_stats['mean_max_fitness'].values
                    grouped_stats_mean_diversity_algo1 = grouped_stats['mean_diversity'].values
                elif algorithm_name == "Algorithm 2":
                    grouped_stats_mean_max_fitness_algo2 = grouped_stats['mean_max_fitness'].values
                    grouped_stats_mean_diversity_algo2 = grouped_stats['mean_diversity'].values

        # Ensure both algorithms' stats are available before running the comparison
        if grouped_stats_mean_max_fitness_algo1 is not None and grouped_stats_mean_max_fitness_algo2 is not None:
            print(f'\nThe statistics for the mean best fitness for enemy {enemy} are:')
            p_fitness = compare_datasets(grouped_stats_mean_max_fitness_algo1, grouped_stats_mean_max_fitness_algo2)

        if grouped_stats_mean_diversity_algo1 is not None and grouped_stats_mean_diversity_algo2 is not None:
            print(f'\nThe statistics for the mean diversity for enemy {enemy} are:')
            p_diversity = compare_datasets(grouped_stats_mean_diversity_algo1, grouped_stats_mean_diversity_algo2)
            

# Main script
if __name__ == "__main__":
    # Process and plot the results for the selected enemies
    algorithm_dfs_dict = process_across_enemies(enemies_to_evaluate, algorithm_dirs)

    # Generate subplots for all enemies with separate fitness and diversity plots
    make_subplots_across_enemies(algorithm_dfs_dict, enemies_to_evaluate)

    # Perform statistical tests
    perform_stats_test(algorithm_dfs_dict, enemies_to_evaluate)