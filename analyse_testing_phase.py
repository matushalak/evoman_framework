###############################################################################
# Authors: Matus Halak, Rick Geling, Rens Koppelman, Isis van Loenen
# matus.halak@gmail.com, rickgeling@gmail.com   
###############################################################################

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# set manual input:  
ALGORITHMS = ["EA1", "EA2"]  # Order matters for order of plotting  
enemy_sets = ['2578', '123578']  # List of enemy sets to process (note: using strings for simplicity) #CHANGED
base_folder = 'gain_res'  # Folder where data is stored  
custom_legend_labels = ["EA1", "EA2"]  


def read_gains(file_path):

    with open(file_path, 'r') as f:
        data = json.load(f)
    
    gains = []  #CHANGED: Changed variable name to `gains` instead of `mean_gains`
    for run_key, gains_per_run in data.items():  #CHANGED: Changed `gains` to `gains_per_run`
        gains.append(gains_per_run)  #CHANGED: Use all gains directly without averaging

    return gains  #CHANGED: Return `gains` instead of `mean_gains`


def gather_data(base_folder, algorithms, enemy_sets):

    data = []

    for enemy_set in enemy_sets:  
        for algo in algorithms:
            file_path = os.path.join(f"{algo}_files", f"{base_folder}_{algo}", f"enemy_{enemy_set}_gains.json")  #CHANGED
            if os.path.exists(file_path):
                gains  = read_gains(file_path)
                #store each mean gain along with the associated algorithm and enemy
                data.extend([[enemy_set, algo, gain] for gain in gains])  #CHANGED: changed `enemy` to `enemy_set`
            else:
                print(f"File not found: {file_path}")
    
    return pd.DataFrame(data, columns=['EnemySet', 'Algorithm', 'Gain']) #return dataframe


def generate_boxplots(df, legend_labels=None):  #CHANGED: Removed p_values and enemies parameters, adjusted to new format
    plt.figure(figsize=(10, 6))  

    # Map the enemy set values to shorter labels
    enemy_set_mapping = {
        '2578': '1',
        '123578': '2'
    }

    # Apply the mapping to df['EnemySet'] to convert to set numbers
    df['EnemySet'] = df['EnemySet'].replace(enemy_set_mapping)

    # Map the algorithm names to more readable labels, and combine with enemy set info for the x-axis
    df['Algorithm_EnemySet'] = df['Algorithm'].replace({
        'EA1': 'Baseline EA',
        'EA2': 'Neat'
    }) + ' (set ' + df['EnemySet'] + ')'

    # Customize colors with alpha adjustments
    base_color = '#ff7f0e'  # Orange
    neat_color = '#d62728'  # Red

    # Customize colors
    palette = {
        "Baseline EA (set 1)": base_color,
        "Neat (set 1)": neat_color,
        "Baseline EA (set 2)": sns.utils.set_hls_values(base_color, l=0.7),
        "Neat (set 2)": sns.utils.set_hls_values(neat_color, l=0.7)
    }
    # create the box plot with customized palette
    ax = sns.boxplot(x='Algorithm_EnemySet', y='Gain', data=df, palette=palette, width=0.6)  #CHANGED: Updated x-axis to be Algorithm_EnemySet

    # remove and right axes spines (open the top and right side of the plot)
    sns.despine(ax=ax, top=True, right=True)  

    # set the labels
    ax.set_xlabel('Evolutionary Algorithms', fontsize=16, labelpad=20)  #CHANGED: Updated x-axis label
    ax.set_ylabel('Gain', fontsize=16)  

    # Set custom legend labels if provided
    if legend_labels is not None:  
        handles, _ = ax.get_legend_handles_labels()  
        ax.legend(handles, legend_labels, loc='lower right', fontsize=14)  
    else:  
        ax.legend(loc='lower right', fontsize=14)  

        # Add a custom legend for the enemy sets
    #custom_legend = [
    #    plt.Line2D([0], [0], color="lightgreen", lw=4, label='Enemy set 1'),
    #    plt.Line2D([0], [0], color="lightblue", lw=4, label='Enemy set 2')
    #]
    #plt.legend(handles=custom_legend, loc='upper right', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=14)


    plt.tight_layout()  
    plt.savefig('box_plot.png')  # Uncomment to save the figure  
    plt.show()  


def perform_statistical_tests(df, enemy_sets, algorithms):

    p_values1 = []
    p_values2 = []
    for enemy_set in enemy_sets:
        algo1_gains = df[(df['EnemySet'] == enemy_set) & (df['Algorithm'] == algorithms[0])]['Gain']
        algo2_gains = df[(df['EnemySet'] == enemy_set) & (df['Algorithm'] == algorithms[1])]['Gain']
        
        #perform t-test (or you can use Mann-Whitney U test if needed)
        t_stat, p_value = stats.ttest_ind(algo1_gains, algo2_gains)
        
        print(f"Enemy Set {enemy_set} - T-test between {algorithms[0]} and {algorithms[1]}: t-stat={t_stat:.4f}, p-value={p_value:.16f}")
        p_values1.append(p_value)

        #perform levene test:
        stat, p_value = stats.levene(algo1_gains, algo2_gains)
        print(f"Enemy Set {enemy_set} - Levene's Test between {algorithms[0]} and {algorithms[1]}, statistic: {stat}, p-value: {p_value}")

    for algo in algorithms:
        algo1_gains2 = df[(df['EnemySet'] == enemy_sets[0]) & (df['Algorithm'] == algo)]['Gain']
        algo2_gains2 = df[(df['EnemySet'] == enemy_sets[1]) & (df['Algorithm'] == algo)]['Gain']
        
        #perform t-test (or you can use Mann-Whitney U test if needed)
        t_stat2, p_value2 = stats.ttest_ind(algo1_gains2, algo2_gains2)
        
        print(f"Algorithm {algo} - T-test between {enemy_sets[0]} and {enemy_sets[1]}: t-stat={t_stat2:.4f}, p-value={p_value2:.16f}")
        p_values2.append(p_value2)

    return p_values1, p_values2


# Function to calculate and print median gain
def print_median_gains(df, algorithms, enemy_sets):
    for enemy_set in enemy_sets:
        for algo in algorithms:
            algo_gains = df[(df['EnemySet'] == enemy_set) & (df['Algorithm'] == algo)]['Gain']
            median_gain = algo_gains.median()
            print(f"Median gain for {algo} on Enemy Set {enemy_set}: {median_gain:.2f}")


#main script
if __name__ == "__main__":

    #step 1: Gather data from JSON files
    df = gather_data(base_folder, ALGORITHMS, enemy_sets)

    #step 2: Perform statistical tests to compare the algorithms
    p_values = perform_statistical_tests(df, enemy_sets, ALGORITHMS)

    #step 3: print the median of the gains
    print_median_gains(df, ALGORITHMS, enemy_sets)

    #step 4: Generate boxplots comparing the algorithms for each enemy and include p-values
    #generate_boxplots(df, p_values, enemy_sets, custom_legend_labels)
    generate_boxplots(df, custom_legend_labels) 



