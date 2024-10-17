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

# set manual input:  #UNCHANGED
ALGORITHMS = ["EA1", "EA2"]  # Order matters for order of plotting  #UNCHANGED
enemy_sets = ['2578', '123578']  # List of enemy sets to process (note: using strings for simplicity) #CHANGED
base_folder = 'gain_res'  # Folder where data is stored  #UNCHANGED
custom_legend_labels = ["EA1", "EA2"]  #UNCHANGED

#now, the script starts from: if __name__ == "__main__"


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

# def add_stat_annotation(ax, p_values, df):

#     y_offsets = [5, 5, 5]  #increasethis value to raise the line higher above the boxes
#     x_shift = 0.25  #control space between boxes
    
#     for i, enemy in enumerate(enemies):
#         algo1_data = df[(df['Enemy'] == enemy) & (df['Algorithm'] == ALGORITHMS[0])]['Gain']
#         algo2_data = df[(df['Enemy'] == enemy) & (df['Algorithm'] == ALGORITHMS[1])]['Gain']
#         y_max = max(max(algo1_data), max(algo2_data))
#         y = y_max + y_offsets[i]  # Raise the p-value line higher than the box
        
#         #draw horizontal line between the two boxes
#         ax.plot([i - x_shift, i + x_shift], [y, y], lw=1.5, color='black')
#         #addthe p-value above the line
#         ax.text(i, y + 0.5, f"p = {p_values[i]:.4f}", ha='center', va='bottom', color='black', fontsize=12)

#     return 0

# def generate_boxplots(df, p_values, enemies, legend_labels=None):

#     plt.figure(figsize=(10, 6))

#     #customize colors
#     palette = {"EA1": "royalblue", "EA2": "lightblue"}
#     #creaete the box plot with customized palette and adjusted box positions
#     ax = sns.boxplot(x='Enemy', y='Gain', hue='Algorithm', data=df, palette=palette, width=0.6)
    
#     #remaining aesthetical operations: 
#     #add statistical annotations in an elegant way (horizontal lines and p-values)
#     add_stat_annotation(ax, p_values, df)
#     #remove and right axes spines (open the top and right side of the plot)
#     sns.despine(ax=ax, top=True, right=True)

#     #set the labels,    TO DO: make label names manual input?
#     ax.set_xticklabels([f'Enemy {enemy}' for enemy in enemies], fontsize=14)
#     ax.set_xlabel('', fontsize=16)
#     ax.set_ylabel('Individual gain', fontsize=16)
#     #ax.set_title('Mean individual gain of every best performing individual out of 10 optimizations', fontsize=15)
#     # Set custom legend labels if provided
#     if legend_labels is not None:
#         handles, _ = ax.get_legend_handles_labels()
#         ax.legend(handles, legend_labels, loc='lower right', fontsize=12)
#     else:
#         ax.legend(loc='lower right', fontsize=12)
    
#     plt.tight_layout()
#     plt.savefig('box_plot.png')  # Uncomment to save the figure
#     plt.show()

def generate_boxplots(df, legend_labels=None):  #CHANGED: Removed p_values and enemies parameters, adjusted to new format
    plt.figure(figsize=(10, 6))  #UNCHANGED

    # Create a new column to combine Algorithm and EnemySet for the x-axis
    df['Algorithm_EnemySet'] = df['Algorithm'] + ' (Enemy set ' + df['EnemySet'] + ')'  #CHANGED: New column for grouping by Algorithm and EnemySet

    # customize colors
    palette = {"EA1 (Enemy set 2578)": "royalblue", 
               "EA2 (Enemy set 2578)": "royalblue", 
               "EA1 (Enemy set 123578)": "lightblue",
               "EA2 (Enemy set 123578)": "lightblue",}  #UNCHANGED

    # create the box plot with customized palette
    ax = sns.boxplot(x='Algorithm_EnemySet', y='Gain', data=df, palette=palette, width=0.6)  #CHANGED: Updated x-axis to be Algorithm_EnemySet

    # remove and right axes spines (open the top and right side of the plot)
    sns.despine(ax=ax, top=True, right=True)  #UNCHANGED

    # set the labels
    ax.set_xlabel('Evolutionary Algorithms', fontsize=16)  #CHANGED: Updated x-axis label
    ax.set_ylabel('Individual Gain', fontsize=16)  #UNCHANGED

    # Set custom legend labels if provided
    if legend_labels is not None:  #UNCHANGED
        handles, _ = ax.get_legend_handles_labels()  #UNCHANGED
        ax.legend(handles, legend_labels, loc='lower right', fontsize=12)  #UNCHANGED
    else:  #UNCHANGED
        ax.legend(loc='lower right', fontsize=12)  #UNCHANGED

    plt.tight_layout()  #UNCHANGED
    plt.savefig('box_plot.png')  # Uncomment to save the figure  #UNCHANGED
    plt.show()  #UNCHANGED


def perform_statistical_tests(df, enemy_sets, algorithms):

    p_values = []
    for enemy_set in enemy_sets:
        algo1_gains = df[(df['EnemySet'] == enemy_set) & (df['Algorithm'] == algorithms[0])]['Gain']
        algo2_gains = df[(df['EnemySet'] == enemy_set) & (df['Algorithm'] == algorithms[1])]['Gain']
        
        #perform t-test (or you can use Mann-Whitney U test if needed)
        t_stat, p_value = stats.ttest_ind(algo1_gains, algo2_gains)
        
        print(f"Enemy Set {enemy_set} - T-test between {algorithms[0]} and {algorithms[1]}: t-stat={t_stat:.4f}, p-value={p_value:.4f}")
        p_values.append(p_value)

    return p_values

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

