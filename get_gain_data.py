
# imports framework
import sys

from evoman.environment import Environment
from demo_controller import player_controller
import os
import json

# Set the base folder for dummy data
base_folder = 'box_plot_gains'
os.makedirs(base_folder, exist_ok=True)
algorithms = ['EA1', 'EA2']
enemies = [2, 6, 8]
n_hidden = 10


# Function to save dummy data in JSON format
def save_json_data(algo_folder, enemy, data):
    os.makedirs(algo_folder, exist_ok=True)
    file_path = os.path.join(algo_folder, f'enemy_{enemy}_gains.json')
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved: {file_path}")



def main():

    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=base_folder,
                    enemies=[enemy],
                    playermode="ai",
                    player_controller=player_controller(n_hidden), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)

    # go through the algorithms:
    #   - first EA1
    #       - EN5: go through all 10 runs, for every run, run the best solution and (alltime or best?)
    #       - save the gain for EA1, EN5 in a dictionary such that it can be stored as a json file using save_json_data()
    #       - then do EN6, then EN8 (in the case we use enemies 5, 6 and 8)
    #  - then do the same for EA1

    # dictionary example:
    # dummy_data_algo2_enemy_2 = {
    # "run_1": [98.80, 74.80, 91.60, 89.80, -17.00],
    # "run_2": [68.80, 86.80, 77.20, 89.20, 87.40],
    # "run_3": [89.20, 94.60, 95.20, 89.20, 88.00],
    # "run_4": [98.80, 78.40, 94.60, 55.00, 88.00],
    # "run_5": [76.60, 88.00, 95.80, 67.60, 67.00],
    # "run_6": [89.20, 73.00, 89.80, 93.40, 58.00],
    # "run_7": [91.60, 92.20, 87.40, 90.40, 89.80],
    # "run_8": [88.00, 90.40, 97.60, 87.40, 84.40],
    # "run_9": [94.60, 82.00, 65.80, 93.40, 93.40],
    # "run_10": [-15.00, 61.00, 86.80, 100.00, 85.60]
    # }
    
    for enemy in enemies:
        for algo in algorithms:
            algo_folder = os.path.join(base_folder, algo)
            
            # Save data for each enemy and algorithm
            if algo == 'algo1':
                if enemy == 2:
                    save_json_data(algo_folder, enemy, dummy_data_algo1_enemy_2)
                elif enemy == 6:
                    save_json_data(algo_folder, enemy, dummy_data_algo1_enemy_6)
                elif enemy == 8:
                    save_json_data(algo_folder, enemy, dummy_data_algo1_enemy_8)
            else:  # algo2
                if enemy == 2:
                    save_json_data(algo_folder, enemy, dummy_data_algo2_enemy_2)
                elif enemy == 6:
                    save_json_data(algo_folder, enemy, dummy_data_algo2_enemy_6)
                elif enemy == 8:
                    save_json_data(algo_folder, enemy, dummy_data_algo2_enemy_8)

if __name__ == "__main__":
    main()
