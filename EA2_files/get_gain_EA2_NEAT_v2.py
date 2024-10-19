###############################################################################
# Authors: Matus Halak, Rick Geling, Rens Koppelman, Isis van Loenen
# matus.halak@gmail.com, rickgeling@gmail.com   
###############################################################################

import sys
from evoman.environment import Environment
from neat_controller import neat_controller
import os
import json
import numpy as np
import pickle as pkl

# Set the base folder for dummy data
base_folder = 'num_beaten_res_EA2'
os.makedirs(base_folder, exist_ok=True)
algorithms = ['NEAT']
folder_names = {
    'NEAT': 'NEAT_final_runs',  # Folder for NEAT
}
config = ['configs_1610_010734/neat_config1.txt', 
          'configs_1610_010734/neat_config2.txt']
enemies = [1, 2, 3, 4, 5, 6, 7, 8]
enemy_sets = ['2578', '123578']
num_runs = 50

# Function to save dummy data in JSON format
def save_json_data(algo_folder, enemy_group, data):
    os.makedirs(algo_folder, exist_ok=True)
    file_path = os.path.join(algo_folder, f'enemy_{enemy_group}_results.json')
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved: {file_path}")

# Runs game (evaluate fitness for 1 individual)
def run_game(env: Environment, individual):
    '''Runs game and returns individual solution's fitness'''
    # vfitness, vplayerlife, venemylife, vtime
    fitness, p, e, t = env.play(pcont=individual)
    return fitness, p, e, t

def main(base_folder, algorithms, enemies, enemy_sets, config, 
         folder_names, num_runs):

    # Choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # Go through the algorithms:
    for algo in algorithms:
        for i, enemy_group in enumerate(enemy_sets):
            # Dictionary to store results for the current algorithm and enemy set
            results_dict = {}
            cfg = config[i]

            # Iterate through the runs
            for run in range(1, num_runs + 1):
                folder_name = folder_names[algo]
                run_folder = os.path.join(folder_name, f'EN{enemy_group}', f'run_{run}_EN{enemy_group}')
                alltime_file = os.path.join(run_folder, 'best.pkl')

                # Check if the best.pkl file exists
                if os.path.exists(alltime_file):
                    with open(alltime_file, 'rb') as f:
                        sol = pkl.load(f)

                    # Initialize results for each enemy
                    results = []

                    # Loop through each enemy
                    for enemy in enemies:
                        # Initialize the environment for the current enemy
                        env = Environment(experiment_name=base_folder,
                                          playermode="ai",
                                          enemies=[enemy],  # Set the current enemy
                                          multiplemode='no',  # Single enemy at a time
                                          player_controller=neat_controller(config_path=cfg),  # Use neat controller
                                          enemymode="static",
                                          level=2,
                                          speed="fastest",
                                          visuals=False)

                        # Play the game and calculate gain
                        fitness, p, e, t = run_game(env, sol)
                        gain = p - e  # Calculate gain as p (player life) - e (enemy life)

                        if gain > 0:
                            results.append(1)  # Beaten
                        elif gain < 0:
                            results.append(0)  # Defeated
                        else:
                            print(f"Gain is zero for run {run}, enemy {enemy}. Breaking.")
                            break

                    # Store the results for the current run
                    results_dict[f'run_{run}'] = results

                else:
                    print(f"File not found: {alltime_file}")
                    current_directory = os.getcwd()
                    print(f"Current Directory: {current_directory}")

            # Save the results for the current algorithm and enemy set
            save_json_data(os.path.join(base_folder), enemy_group, results_dict)

if __name__ == "__main__":
    main(base_folder, algorithms, enemies, enemy_sets, config, 
         folder_names, num_runs)
