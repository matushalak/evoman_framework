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
base_folder = 'gain_res_EA2'
os.makedirs(base_folder, exist_ok=True)
algorithms = ['EA2']#, 'EA2']
folder_names = {
    'EA2': 'NEAT_final_runs',  # Folder for EA1
}
config = [f'configs_1610_010734/neat_config1.txt', 
          f'configs_1610_010734/neat_config2.txt']
enemies = [1, 2, 3, 4, 5, 6, 7, 8]  
enemy_sets = ['2578','123578']
num_runs = 50

#TO CHANGE:
# load in the config file                       CHECK
# player_controller --> neat_controller         CHECK


# Function to save dummy data in JSON format
def save_json_data(algo_folder, enemy_group, data):
    os.makedirs(algo_folder, exist_ok=True)
    file_path = os.path.join(algo_folder, f'enemy_{enemy_group}_gains.json')
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved: {file_path}")


# # runs game (evaluate fitness for 1 individual)
def run_game(env:Environment, individual):
    '''Runs game and returns individual solution's fitness'''
    
    # vfitness, vplayerlife, venemylife, vtime
    fitness, p, e, t = env.play(pcont=individual)

    return fitness, p, e, t

def main(base_folder, algorithms, enemies, enemy_sets, config, 
         folder_names, num_runs):

    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # # initializes simulation in individual evolution mode, for single static enemy.
    # env = Environment(experiment_name=base_folder,
    #                 playermode="ai",
    #                 enemies=enemies,  # Unchanged
    #                 multiplemode='yes',  # Unchanged
    #                 player_controller=neat_controller(config), # you can insert your own controller here
    #                 enemymode="static",
    #                 level=2,
    #                 speed="fastest",
    #                 visuals=False)

    # Go through the algorithms:
    for algo in algorithms:
        for i, enemy_group in enumerate(enemy_sets):
            #dictionary to store gains for the current algorithm and enemy
            gains_dict = {}
            cfg = config[i]

            env = Environment(experiment_name=base_folder,
                            playermode="ai",
                            enemies=enemies,  # Unchanged
                            multiplemode='yes',  # Unchanged
                            player_controller=neat_controller(config_path=cfg), # you can insert your own controller here
                            enemymode="static",
                            level=2,
                            speed="fastest",
                            visuals=False)

            #Update the enemy
            #env.update_parameter('enemies',[enemy])

            # Iterate through the runs
            for run in range(1, num_runs+1):
                folder_name = folder_names[algo]
                run_folder = os.path.join(folder_name, f'EN{enemy_group}', f'run_{run}_EN{enemy_group}')
                alltime_file = os.path.join(run_folder, 'best.pkl')

                
                #check if the alltime.txt file exists
                if os.path.exists(alltime_file):
                    with open(alltime_file, 'rb') as f:
                        sol = pkl.load(f)

                    fitness, p, e, t = env.play(pcont = sol)
                    gain = p - e  # Calculate gain as p (player life) - e (enemy life)

                    #store the 5 gains for this run in the dictionary
                    gains_dict[f'run_{run}'] = gain

                else:
                    print(f"File not found: {alltime_file}")
                    current_directory = os.getcwd()
                    print(f"Current Directory: {current_directory}")

            #save the gains for the current algorithm and enemy
            save_json_data(os.path.join(base_folder), enemy_group, gains_dict)


if __name__ == "__main__":
    main(base_folder, algorithms, enemies, enemy_sets, config, 
         folder_names, num_runs)
