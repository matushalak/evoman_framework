###############################################################################
# Authors: Matus Halak, Rick Geling, Rens Koppelman, Isis van Loenen
# matus.halak@gmail.com, rickgeling@gmail.com   
###############################################################################

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, ttest_rel, wilcoxon
import numpy as np
from statsmodels.stats.stattools import durbin_watson

##### ------------------------------------------ NOTE: --> works for Task 2 EA1 now, not for neat yet!
#       in 'enemies_to_evaluate' the enemy sets can be selected
#       in 'algorithm_dirs' the root folder name (where the multiple runs are saved) should be set
#       'max_gen' can be set to any number upto the number op gens that where used to get the data 


##### ------------------------------------------ TODO: 
#       make this process streamlined for the way (& location where) neat output is saved
#       change line +/-52: pass_folder = os.path.join('EA1_files', base_folder) --> only for EA1, not robust
#       find replacement for 'if algorithm == 'Classic EA':' and 'elif algorithm == 'IM-EA':'
#       maybe include a parser like in the other scripts we built


#set manual input for enemies to evaluate (e.g., [5, 6])
enemies_to_evaluate = [[2, 3],[5, 6]]

#base directories for both algorithms
algorithm_dirs = {
    'Classic EA': 'TEST_EA1_line_plot_runs',
    #'Neat': 'NEAT_report_results'
}

# Set the maximum generation up to where to want to plot and do statistical tests
max_gen = 50 # To limit the number of generations shown in the plot
plot_titles = ['A', 'B'] # Just for plotting aestetics  


def process_across_enemies(enemy_sets, algorithm_dirs, max_gen=None):
    """
    Process results for all enemies and both algorithms, and generate n x 2 subplots for fitness and diversity.
    """
    algorithm_dfs_dict = {}

    for enemy_group in enemy_sets:
        print(f"Processing enemy {enemy_group}...")

        #dictionary to hold the dataframes for both algorithms
        algorithm_dfs = {}

        #process results for both algorithms
        for algorithm_name, base_folder in algorithm_dirs.items():
            pass_folder = os.path.join('EA1_files', base_folder)
            df = process_results_for_enemy(pass_folder, enemy_group, algorithm_name, max_gen)
            algorithm_dfs[algorithm_name] = df
        
        #store the results for this enemy
        algorithm_dfs_dict[str(enemy_group)] = algorithm_dfs

    return algorithm_dfs_dict



def process_results_for_enemy(base_folder, enemy_group, algorithm, max_gen=None):
    """
    Process the results for a given enemy for one algorithm, handling different logging formats.
    """
    enemy_folder = os.path.join(base_folder, f"EN{enemy_group}")
    
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
                #read results file
                if algorithm == 'Classic EA':
                    #for Algorithm 1
                    df = pd.read_csv(result_file, delim_whitespace=True)
                    #drop the first duplicate generation 0 row (if it exists)
                    df = df.drop_duplicates(subset=['gen'], keep='last')
                    
                elif algorithm == 'IM-EA':
                    #for Algorithm 2
                    df = pd.read_csv(result_file, delim_whitespace=True, comment='#', header=None)
                    df.columns = ['gen', 'best', 'mean', 'std', 'diversity']  # Add column names
                    #convert the 'gen' column to integer since it's stored as float
                    df['gen'] = df['gen'].astype(int)
                    
                # filter rows based on max generation if max_gen is set
                if max_gen is not None:
                    df = df[df['gen'] <= max_gen]
                # ensure valid numeric data
                df = df[pd.to_numeric(df['best'], errors='coerce').notnull()]

                # Convert columns to numeric
                df['best'] = pd.to_numeric(df['best'])
                df['mean'] = pd.to_numeric(df['mean'])
                df['std'] = pd.to_numeric(df['std'])
                df['gen'] = pd.to_numeric(df['gen'])

                # Append DataFrame to the list
                all_dfs.append(df)

    if all_dfs:
        return pd.concat(all_dfs)  # Return the concatenated dataframe
    return None


def make_subplots_across_enemies(algorithm_dfs_dict, enemy_sets, plot_titles):
    """
    Create subplots with the results for all enemy_set, with separate plots for fitness and diversity.
    """
    n_enemy_sets = len(enemy_sets)  # Number of enemy_set provided
    fig, axes = plt.subplots(nrows=n_enemy_sets, ncols=1, figsize=(12, 18))  # Create an n x 2 grid

    if n_enemy_sets == 1:
        axes = [axes]  #handle case where there's only one enemy to plot

    #iterate over each enemy and corresponding axes
    for i, (enemy_group, ax_pair) in enumerate(zip(enemy_sets, axes)):
        ax_fitness = ax_pair  # Unpack the axes for fitness and diversity

        algorithm_dfs = algorithm_dfs_dict[str(enemy_group)]

        for algorithm_name, df in algorithm_dfs.items():
            if df is not None:
                #group by generation and calculate mean and std for mean and best fitness
                grouped_stats = df.groupby('gen').agg(
                    mean_avg_fitness=('mean', 'mean'),
                    std_avg_fitness=('mean', 'std'),
                    mean_max_fitness=('best', 'mean'),
                    std_max_fitness=('best', 'std')
                )

                generations = grouped_stats.index

                #plot mean fitness with standard deviation shading on fitness axis
                ax_fitness.plot(generations, grouped_stats['mean_avg_fitness'], label=f"{algorithm_name}: mean fitness", 
                                linestyle='--', linewidth=2)
                ax_fitness.fill_between(generations,
                                        grouped_stats['mean_avg_fitness'] - grouped_stats['std_avg_fitness'],
                                        grouped_stats['mean_avg_fitness'] + grouped_stats['std_avg_fitness'],
                                        alpha=0.2)

                #plot best fitness with standard deviation shading on fitness axis
                ax_fitness.plot(generations, grouped_stats['mean_max_fitness'], label=f"{algorithm_name}: best fitness", 
                                linestyle='-', linewidth=2)
                ax_fitness.fill_between(generations,
                                        grouped_stats['mean_max_fitness'] - grouped_stats['std_max_fitness'],
                                        grouped_stats['mean_max_fitness'] + grouped_stats['std_max_fitness'],
                                        alpha=0.2)

        #customize the fitness plot (left)
        ax_fitness.set_title(f'{plot_titles[i]}) Fitness evolution for enemy set {enemy_group}', fontsize=18)
        ax_fitness.set_xlabel('Generation', fontsize=16)
        ax_fitness.set_ylabel('Fitness', fontsize=16)
        ax_fitness.grid(True)

        ax_fitness.tick_params(axis='both', which='major', labelsize=12)  # Change '10' to your desired font size


        #combine legends for fitness and diversity
        fitness_lines, fitness_labels = ax_fitness.get_legend_handles_labels()
        ax_fitness.legend(fitness_lines, fitness_labels, loc='lower right', fontsize=14)

    #adjust layout to prevent overlap between subplots
    plt.tight_layout(pad=3.0)
    plt.savefig('line_plots.png')  # Uncomment to save the figure
    plt.show()


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

        #initialize placeholders for the mean max fitness and mean diversity of both algorithms
        grouped_stats_mean_max_fitness_algo1 = None
        grouped_stats_mean_max_fitness_algo2 = None
        grouped_stats_mean_diversity_algo1 = None
        grouped_stats_mean_diversity_algo2 = None

        #iterate through algorithms and save the stats for both algorithms
        for algorithm_name, df in algorithm_dfs.items():
            if df is not None:
                #group by generation and calculate mean max fitness and mean diversity
                grouped_stats = df.groupby('gen').agg(
                    mean_max_fitness=('best', 'mean'),
                    mean_diversity=('diversity', 'mean')
                )

                #store stats for the correct algorithm
                if algorithm_name == "Classic EA":
                    grouped_stats_mean_max_fitness_algo1 = grouped_stats['mean_max_fitness'].values
                    grouped_stats_mean_diversity_algo1 = grouped_stats['mean_diversity'].values
                elif algorithm_name == "IM-EA":
                    grouped_stats_mean_max_fitness_algo2 = grouped_stats['mean_max_fitness'].values
                    grouped_stats_mean_diversity_algo2 = grouped_stats['mean_diversity'].values

        sum_difference_best = sum(grouped_stats_mean_max_fitness_algo1-grouped_stats_mean_max_fitness_algo2)
        print(f'\nSum of the difference between the mean best fitnesses of EA1 and EA2: ' 
              f'{sum_difference_best}')

        #ensure both algorithms' stats are available before running the comparison
        if grouped_stats_mean_max_fitness_algo1 is not None and grouped_stats_mean_max_fitness_algo2 is not None:
            print(f'\nThe statistics for the mean best fitness for enemy {enemy} are:')
            p_fitness = compare_datasets(grouped_stats_mean_max_fitness_algo1, grouped_stats_mean_max_fitness_algo2)

        if grouped_stats_mean_diversity_algo1 is not None and grouped_stats_mean_diversity_algo2 is not None:
            print(f'\nThe statistics for the mean diversity for enemy {enemy} are:')
            p_diversity = compare_datasets(grouped_stats_mean_diversity_algo1, grouped_stats_mean_diversity_algo2)
            

# Main script
if __name__ == "__main__":
    #step 1: rocess and plot the results for the selected enemies
    algorithm_dfs_dict = process_across_enemies(enemies_to_evaluate, algorithm_dirs, max_gen)

    #step 2: enerate subplots for all enemies with separate fitness and diversity plots
    make_subplots_across_enemies(algorithm_dfs_dict, enemies_to_evaluate, plot_titles)

    #step 3: perform statistical tests
    #perform_stats_test(algorithm_dfs_dict, enemies_to_evaluate)