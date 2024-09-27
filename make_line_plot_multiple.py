import os
import pandas as pd
import matplotlib.pyplot as plt

# Set manual input for enemies to evaluate (e.g., [5, 6])
enemies_to_evaluate = [6]

# Base directories for both algorithms
algorithm_dirs = {
    'Algorithm 1': 'EA1_line_plot_runs',
    'Algorithm 2': 'EA2_line_plot_runs'
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
                df['gen'] = pd.to_numeric(df['gen'])

                # Append DataFrame to the list
                all_dfs.append(df)

    if all_dfs:
        return pd.concat(all_dfs)  # Return the concatenated dataframe
    return None

def make_subplots_across_enemies(algorithm_dfs_dict, enemies):
    """
    Create subplots with the results for all enemies.
    """
    n_enemies = len(enemies)  # Number of enemies provided
    fig, axes = plt.subplots(nrows=n_enemies, ncols=1, figsize=(10, 6 * n_enemies))

    if n_enemies == 1:
        axes = [axes]  # Handle the case where there's only one enemy to plot

    # Iterate over each enemy and corresponding axis
    for i, (enemy, ax) in enumerate(zip(enemies, axes)):
        algorithm_dfs = algorithm_dfs_dict[enemy]

        for algorithm_name, df in algorithm_dfs.items():
            if df is not None:
                # Group by generation and calculate mean and std for mean and best fitness
                grouped_stats = df.groupby('gen').agg(
                    mean_avg_fitness=('mean', 'mean'),
                    std_avg_fitness=('mean', 'std'),
                    mean_max_fitness=('best', 'mean'),
                    std_max_fitness=('best', 'std')
                )

                generations = grouped_stats.index

                # Plot mean fitness with standard deviation shading
                ax.plot(generations, grouped_stats['mean_avg_fitness'], label=f"{algorithm_name} mean fitness", linestyle='--')
                ax.fill_between(generations,
                                grouped_stats['mean_avg_fitness'] - grouped_stats['std_avg_fitness'],
                                grouped_stats['mean_avg_fitness'] + grouped_stats['std_avg_fitness'],
                                alpha=0.2)

                # Plot best fitness with standard deviation shading
                ax.plot(generations, grouped_stats['mean_max_fitness'], label=f"{algorithm_name} best fitness", linestyle='-')
                ax.fill_between(generations,
                                grouped_stats['mean_max_fitness'] - grouped_stats['std_max_fitness'],
                                grouped_stats['mean_max_fitness'] + grouped_stats['std_max_fitness'],
                                alpha=0.2)

        # Customize each subplot
        ax.set_title(f'Fitness evolution for enemy {enemy}', fontsize=14)
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Fitness', fontsize=12)
        ax.grid(True)
        ax.legend()

    # Adjust layout and show the full figure
    plt.tight_layout()
    plt.show()

def process_and_plot(enemies_to_evaluate, algorithm_dirs):
    """
    Process results for all enemies and both algorithms, and generate plots.
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

    # Generate subplots for all enemies
    make_subplots_across_enemies(algorithm_dfs_dict, enemies_to_evaluate)

# Main script
if __name__ == "__main__":
    # Process and plot the results for the selected enemies
    process_and_plot(enemies_to_evaluate, algorithm_dirs)
