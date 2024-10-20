###############################################################################
# Authors: Matus Halak, Rick Geling, Rens Koppelman, Isis van Loenen
# matus.halak@gmail.com, rickgeling@gmail.com   
###############################################################################

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, ttest_rel, wilcoxon, f_oneway
import numpy as np
from statsmodels.stats.stattools import durbin_watson
import seaborn as sns

##### ------------------------------------------ NOTE:
#       in 'enemies_to_evaluate' the enemy sets can be selected
#       in 'algorithm_dirs' the root folder name (where the multiple runs are saved) should be set


#set manual input for enemies to evaluate (e.g., [5, 6])
enemies_to_evaluate = ['2578','123578']

#base directories for both algorithms
algorithm_dirs = {
    'Baseline EA': 'EA1_final_runs',
    'Neat': 'NEAT_final_runs'
}

max_gen = None #Don't use this.
plot_titles = ['A', 'B', 'C', 'D'] # Just for plotting aestetics  


# ------------------------------ Data processing function(s) ------------------------------ 

def process_across_enemies(enemy_sets, algorithm_dirs, max_gen=None):
    """
    Process results for all enemies and both algorithms, and generate n x 2 subplots for fitness and diversity.
    """
    algorithm_dfs_dict = {}
    max_best_values_dict = {}  # Dictionary to hold max 'best' values for all enemies and algorithms

    for enemy_group in enemy_sets:
        print(f"Processing enemy {enemy_group}...")

        #dictionary to hold the dataframes for both algorithms
        algorithm_dfs = {}
        enemy_max_best_values = {}  # Dictionary to hold max values for this enemy

        #process results for both algorithms
        for algorithm_name, base_folder in algorithm_dirs.items():
            # Set the appropriate pass_folder based on the algorithm
            if algorithm_name == 'Baseline EA':
                pass_folder = os.path.join('EA1_files', base_folder)
            elif algorithm_name == 'Neat':
                pass_folder = os.path.join('EA2_files', base_folder)
            df, max_best_values  = process_results_for_enemy_group(pass_folder, enemy_group, algorithm_name, max_gen)
            algorithm_dfs[algorithm_name] = df
            enemy_max_best_values[algorithm_name] = max_best_values
        
        #store the results for this enemy
        algorithm_dfs_dict[str(enemy_group)] = algorithm_dfs
        max_best_values_dict[str(enemy_group)] = enemy_max_best_values

    return algorithm_dfs_dict, max_best_values_dict

def process_results_for_enemy_group(base_folder, enemy_group, algorithm, max_gen=None):
    """
    Process the results for a given enemy for one algorithm, handling different logging formats.
    """
    enemy_folder = os.path.join(base_folder, f"EN{enemy_group}")
    
    if not os.path.isdir(enemy_folder):
        print(f"Enemy folder not found: {enemy_folder}")
        return None

    all_dfs = []  # Initialize an empty list to hold dataframes
    max_best_values = {}  # Dictionary to hold max 'best' values for each run

    # Iterate through the run folders (e.g., run_1_ENX, run_2_ENX, etc.)
    for run_folder in os.listdir(enemy_folder):
        run_path = os.path.join(enemy_folder, run_folder)
        
        if os.path.isdir(run_path):
            result_file = os.path.join(run_path, 'results.txt')  # Each run has a 'results.txt' file
            
            if os.path.exists(result_file):
                #read results file
                if algorithm == 'Baseline EA':
                    #for Algorithm 1
                    df = pd.read_csv(result_file, delim_whitespace=True)
                    #drop the first duplicate generation 0 row (if it exists)
                    df = df.drop_duplicates(subset=['gen'], keep='last')
                    
                elif algorithm == 'Neat':
                    # For Algorithm 2
                    df = pd.read_csv(result_file, delimiter=',')  # Read the file using comma as a delimiter
                    df.columns = ['gen', 'best', 'mean', 'std']  # Add column names
                    # Convert the 'gen' column to integer since it's stored as float
                    df['gen'] = df['gen'].astype(int)
                    
                # filter rows based on max generation if max_gen is set
                #if max_gen is not None:
                #    df = df[df['gen'] <= max_gen]
                # ensure valid numeric data
                df = df[pd.to_numeric(df['best'], errors='coerce').notnull()]

                # Convert columns to numeric
                df['best'] = pd.to_numeric(df['best'])
                df['mean'] = pd.to_numeric(df['mean'])
                df['std'] = pd.to_numeric(df['std'])
                df['gen'] = pd.to_numeric(df['gen'])

                max_best = df['best'].max()
                max_best_values[run_folder] = max_best

                # Append DataFrame to the list
                all_dfs.append(df)

    if all_dfs:
        return pd.concat(all_dfs), max_best_values  # Return the concatenated dataframe


# ------------------------------ Plot function ------------------------------ 

def make_subplots_across_enemies(algorithm_dfs_dict, max_best_values_dict, enemy_sets, plot_titles):
    """
    Create subplots with the results for all enemy_set, with separate plots for fitness and diversity.
    """
    n_enemy_sets = len(enemy_sets)  # Number of enemy_set provided
    fig, axes = plt.subplots(nrows=n_enemy_sets, ncols=2, figsize=(14, 14), 
                             gridspec_kw={'width_ratios': [3, 2]})  # Create an n x 2 grid

    if n_enemy_sets == 1:
        axes = [axes]  #handle case where there's only one enemy to plot

    colors = {
    'Baseline EA': "#ff7f0e",
    'Neat': "#d62728"
    }

    #iterate over each enemy and corresponding axes
    for i, (enemy_group, ax_pair) in enumerate(zip(enemy_sets, axes)):
        ax_fitness, ax_violin = ax_pair 

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
                                linestyle='-', linewidth=2)#, color=colors[algorithm_name])
                ax_fitness.fill_between(generations,
                                        grouped_stats['mean_max_fitness'] - grouped_stats['std_max_fitness'],
                                        grouped_stats['mean_max_fitness'] + grouped_stats['std_max_fitness'],
                                        alpha=0.2)

        #customize the fitness plot (left)
        ax_fitness.set_title(f'{plot_titles[i]}) Fitness evolution for enemy set {i+1}', fontsize=18)
        ax_fitness.set_xlabel('Generation', fontsize=16)
        ax_fitness.set_ylabel('Fitness', fontsize=16)
        ax_fitness.set_ylim(-85, 70) 
        ax_fitness.grid(True)

        ax_fitness.tick_params(axis='both', which='major', labelsize=16) 


        #combine legends for fitness and diversity
        fitness_lines, fitness_labels = ax_fitness.get_legend_handles_labels()
        ax_fitness.legend(fitness_lines, fitness_labels, loc='lower right', fontsize=16)

        # Prepare data for the violin plot
        max_best_values = max_best_values_dict[str(enemy_group)]
        plot_data = []
        for algorithm_name, run_values in max_best_values.items():
            for run, max_best_value in run_values.items():
                plot_data.append({
                    'Algorithm': algorithm_name,
                    'Max_Best_Value': max_best_value
                })

        plot_df = pd.DataFrame(plot_data)

        # Plot violin distribution of max_best_values on the right axis
        sns.violinplot(ax=ax_violin, x='Algorithm', y='Max_Best_Value', data=plot_df, split=True, 
                       inner='quartile', palette=["#ff7f0e", "#d62728"]) 

        # Customize the violin plot
        ax_violin.set_title(f'{plot_titles[i + 2]}) Best fitness dist. for enemy set {i+1}', fontsize=18)  
        ax_violin.set_xlabel('Algorithm', fontsize=16)
        ax_violin.set_ylabel('Best Fitness', fontsize=16)
        ax_violin.set_ylim(0, 90)  
        ax_violin.grid(True)

        ax_violin.tick_params(axis='both', which='major', labelsize=16)

    #adjust layout to prevent overlap between subplots
    plt.tight_layout(pad=3.0)
    plt.savefig('line_plots.png')  # Uncomment to save the figure
    plt.show()


# ------------------------------ Stats test function(s) PART 1 ------------------------------ 

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

def levene_test(data1, data2):
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

def f_test(data1, data2):
    """
    Test if two datasets have equal variances using F-test.
    
    Parameters:
    The two datasets to test.
    
    Returns:
    The p-value of F-test test.
    """
    f_stat, p_value = f_oneway(data1, data2)
    print(f"F-test Statistic: {f_stat}, P-value: {p_value}")
    return p_value


# ------------------------------ Stats test function(s) PART 2 ------------------------------ 

def compare_datasets_line_plot(data1, data2):
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
    
def compare_datasets_max_best(data1, data2):
    """
    Compare the variance two datasets using F-test or Levene test based on normality.
    
    Parameters:
    The two datasets to compare
    
    Returns:
    The test used and the p-value.
    """
    # Test for normality
    p_value_normal1 = test_normality(data1)
    p_value_normal2 = test_normality(data2)
    
    # Decide on the test to use
    if p_value_normal1 > 0.05 and p_value_normal2 > 0.05:
        # Normal: F-test
        p_value = f_test(data1, data2)
        print(f"F-test used for max best. P-value: {p_value}")
        return "F-test", p_value
    else:
        # Non-normal: Levene test
        p_value = levene_test(data1, data2)
        print(f"Levene Test used for max best. P-value: {p_value}")
        return "Levene Test", p_value

# ------------------------------ Stats test function(s) PART 3 ------------------------------ 

def perform_stats_test(algorithm_dfs_dict, max_best_values_dict, enemies):
    """
    Perform statistical tests on the mean best fitness and max best values for each enemy set and both algorithms.
    """
    for enemy in enemies:
        print(f"\nProcessing statistical tests for enemy {enemy}...")

        algorithm_dfs = algorithm_dfs_dict[str(enemy)]
        max_best_values = max_best_values_dict[str(enemy)]

        #Initialize placeholders for the mean max fitness and diversity of both algorithms
        grouped_stats_mean_max_fitness_algo1 = None
        grouped_stats_mean_max_fitness_algo2 = None

        #Iterate through algorithms and save the stats for both algorithms
        for algorithm_name, df in algorithm_dfs.items():
            if df is not None:
                #Group by generation and calculate mean max fitness
                grouped_stats = df.groupby('gen').agg(
                    mean_max_fitness=('best', 'mean')
                )

                #Store stats for the correct algorithm
                if algorithm_name == "Baseline EA":
                    grouped_stats_mean_max_fitness_algo1 = grouped_stats['mean_max_fitness'].values
                elif algorithm_name == "Neat":
                    grouped_stats_mean_max_fitness_algo2 = grouped_stats['mean_max_fitness'].values

        #Perform Wilcoxon test to compare mean best fitness between EA1 and NEAT for the enemy set
        if grouped_stats_mean_max_fitness_algo1 is not None and grouped_stats_mean_max_fitness_algo2 is not None:
            print(f'\nPerforming statistical tests for mean best fitness for enemy set {enemy}...')
            test_result, p_value = compare_datasets_line_plot(grouped_stats_mean_max_fitness_algo1, grouped_stats_mean_max_fitness_algo2)

            #Compare mean best fitness between algorithms in the final generation (generation 100)
            print("\nComparing the final generation's best fitness values:")
            final_gen_fitness_algo1 = grouped_stats_mean_max_fitness_algo1[-1]
            final_gen_fitness_algo2 = grouped_stats_mean_max_fitness_algo2[-1]
            final_test_result, final_p_value = wilcoxon([final_gen_fitness_algo1], [final_gen_fitness_algo2])
            print(f"Wilcoxon Test on final generation: Test statistic = {final_test_result}, P-value = {final_p_value}")

        #Collect max best values for both algorithms
        max_best_values_algo1 = []
        max_best_values_algo2 = []

        for algorithm_name, run_values in max_best_values.items():
            if algorithm_name == "Baseline EA":
                max_best_values_algo1 = list(run_values.values())
            elif algorithm_name == "Neat":
                max_best_values_algo2 = list(run_values.values())

        # Perform the comparison for max best values (violin plots)
        if max_best_values_algo1 and max_best_values_algo2:
            print(f'\nPerforming statistical tests for max best values for enemy set {enemy}...')
            compare_datasets_max_best(max_best_values_algo1, max_best_values_algo2)

            #Report variances for enemy set 2 explicitly
            if str(enemy) == '123578':
                variance_algo1 = np.var(max_best_values_algo1)
                variance_algo2 = np.var(max_best_values_algo2)
                print(f"\nVariance for Enemy Set 2 - Baseline EA: {variance_algo1:.4f}")
                print(f"Variance for Enemy Set 2 - Neat: {variance_algo2:.4f}")

            #erform Wilcoxon tests between EA1 and NEAT for Enemy Set 2
            if str(enemy) == '123578':
                test_result, p_value = wilcoxon(max_best_values_algo1, max_best_values_algo2)
                print(f"\nWilcoxon Test for Enemy Set 2 - Baseline EA vs Neat: Test statistic = {test_result}, P-value = {p_value}")

            #Perform Wilcoxon tests between EA1 and NEAT for Enemy Set 1
            elif str(enemy) == '2578':
                test_result, p_value = wilcoxon(max_best_values_algo1, max_best_values_algo2)
                print(f"\nWilcoxon Test for Enemy Set 1 - Baseline EA vs Neat: Test statistic = {test_result}, P-value = {p_value}")

            #Compare the generalizability (gain values) between Enemy Set 1 and Enemy Set 2
            print(f"\nPerforming Wilcoxon tests for generalizability of gains between enemy sets for Baseline EA and NEAT...")
            test_result_1, p_value_1 = wilcoxon(max_best_values_algo1, max_best_values_algo2)
            print(f"Wilcoxon Test for Baseline EA Enemy Set 1 vs. Enemy Set 2: Test statistic = {test_result_1}, P-value = {p_value_1}")
            test_result_2, p_value_2 = wilcoxon(max_best_values_algo1, max_best_values_algo2)
            print(f"Wilcoxon Test for Neat Enemy Set 1 vs. Enemy Set 2: Test statistic = {test_result_2}, P-value = {p_value_2}")

# Main steps
if __name__ == "__main__":
    # Step 1: Process and plot the results for the selected enemies
    algorithm_dfs_dict, max_best_values_dict = process_across_enemies(enemies_to_evaluate, algorithm_dirs, max_gen)

    # Step 2: Generate subplots for all enemies with separate fitness and diversity plots
    make_subplots_across_enemies(algorithm_dfs_dict, max_best_values_dict, enemies_to_evaluate, plot_titles)

    # Step 3: Perform statistical tests as specified
    perform_stats_test(algorithm_dfs_dict, max_best_values_dict, enemies_to_evaluate)
