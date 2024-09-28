import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#set manual input:
ALGORITHMS = ["EA1", "EA2"] #Order matters for order of plotting
ENEMIES = [5, 6, 8]  # List of enemies to process
base_folder = 'box_plot_gains'  # Folder where data is stored

#now, the script starts from: if __name__ == "__main__"


def read_gains(file_path):

    with open(file_path, 'r') as f:
        data = json.load(f)
    
    mean_gains = []
    for run_key, gains in data.items():
        #calc the mean for each run (5 gains per run)
        mean_gains.append(sum(gains) / len(gains))
    
    return mean_gains

def gather_data(base_folder, algorithms, enemies):

    data = []

    for enemy in enemies:
        for algo in algorithms:
            file_path = os.path.join(base_folder, algo, f"enemy_{enemy}_gains.json")
            if os.path.exists(file_path):
                mean_gains = read_gains(file_path)
                #store each mean gain along with the associated algorithm and enemy
                data.extend([[enemy, algo, gain] for gain in mean_gains])
            else:
                print(f"File not found: {file_path}")
    
    return pd.DataFrame(data, columns=['Enemy', 'Algorithm', 'Gain']) #return dataframe

def add_stat_annotation(ax, p_values, df):

    y_offsets = [5, 5, 5]  #increasethis value to raise the line higher above the boxes
    x_shift = 0.25  #control space between boxes
    
    for i, enemy in enumerate(ENEMIES):
        algo1_data = df[(df['Enemy'] == enemy) & (df['Algorithm'] == ALGORITHMS[0])]['Gain']
        algo2_data = df[(df['Enemy'] == enemy) & (df['Algorithm'] == ALGORITHMS[1])]['Gain']
        y_max = max(max(algo1_data), max(algo2_data))
        y = y_max + y_offsets[i]  # Raise the p-value line higher than the box
        
        #draw horizontal line between the two boxes
        ax.plot([i - x_shift, i + x_shift], [y, y], lw=1.5, color='black')
        #addthe p-value above the line
        ax.text(i, y + 0.5, f"p = {p_values[i]:.4f}", ha='center', va='bottom', color='black')

    return 0

def generate_boxplots(df, p_values, ENEMIES):

    plt.figure(figsize=(10, 6))

    #customize colors
    palette = {"EA1": "royalblue", "EA2": "lightblue"}
    #creaete the box plot with customized palette and adjusted box positions
    ax = sns.boxplot(x='Enemy', y='Gain', hue='Algorithm', data=df, palette=palette, width=0.6)
    
    #remaining aesthetical operations: 
    #add statistical annotations in an elegant way (horizontal lines and p-values)
    add_stat_annotation(ax, p_values, df)
    #remove and right axes spines (open the top and right side of the plot)
    sns.despine(ax=ax, top=True, right=True)

    #set the labels,    TO DO: make label names manual input?
    ax.set_xticklabels([f'Enemy {enemy}' for enemy in ENEMIES], fontsize=8)
    ax.set_xlabel('Experiment name', fontsize=12)
    ax.set_ylabel('Individual gain', fontsize=12)
    
    plt.show()

def perform_statistical_tests(df, enemies, algorithms):

    p_values = []
    for enemy in enemies:
        algo1_gains = df[(df['Enemy'] == enemy) & (df['Algorithm'] == algorithms[0])]['Gain']
        algo2_gains = df[(df['Enemy'] == enemy) & (df['Algorithm'] == algorithms[1])]['Gain']
        
        #perform t-test (or you can use Mann-Whitney U test if needed)
        t_stat, p_value = stats.ttest_ind(algo1_gains, algo2_gains)
        
        print(f"Enemy {enemy} - T-test between {algorithms[0]} and {algorithms[1]}: t-stat={t_stat:.4f}, p-value={p_value:.4f}")
        p_values.append(p_value)

    return p_values

#main script
if __name__ == "__main__":

    #step 1: Gather data from JSON files
    df = gather_data(base_folder, ALGORITHMS, ENEMIES)

    #step 2: Perform statistical tests to compare the algorithms
    p_values = perform_statistical_tests(df, ENEMIES, ALGORITHMS)

    #step 3: Generate boxplots comparing the algorithms for each enemy and include p-values
    generate_boxplots(df, p_values, ENEMIES)
