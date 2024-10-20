###############################################################################
# Authors: Matus Halak, Rick Geling, Rens Koppelman, Isis van Loenen
# matus.halak@gmail.com, rickgeling@gmail.com   
###############################################################################

import sys

from evoman.environment import Environment
from demo_controller import player_controller
import os
import json
import numpy as np

# Set the base folder for dummy data
base_folder = 'gain_res_EA1'
os.makedirs(base_folder, exist_ok=True)
algorithms = ['EA1']#, 'EA2']
folder_names = {
    'EA1': 'EA1_final_runs',  # Folder for EA1
}
enemies = [1, 2, 3, 4, 5, 6, 7, 8]  
enemy_sets = ['2578','123578']
n_hidden = 10
num_runs = 50


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
    fitness,p,e,t = env.play(pcont=individual)

    return fitness,p,e,t

def main(base_folder, algorithms, enemies, enemy_sets, n_hidden, 
         folder_names, num_runs):

    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=base_folder,
                    playermode="ai",
                    enemies=enemies,  # Unchanged
                    multiplemode='yes',  # Unchanged
                    player_controller=player_controller(n_hidden), # you can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)

    # Go through the algorithms:
    for algo in algorithms:
        for enemy_group in enemy_sets:
            #dictionary to store gains for the current algorithm and enemy
            gains_dict = {}

            #Update the enemy
            #env.update_parameter('enemies',[enemy])

            # Iterate through the runs
            for run in range(1, num_runs+1):
                folder_name = folder_names[algo]
                run_folder = os.path.join(folder_name, f'EN{enemy_group}', f'run_{run}_EN{enemy_group}')
                alltime_file = os.path.join(run_folder, 'alltime.txt')
                
                #check if the alltime.txt file exists
                if os.path.exists(alltime_file):
                    sol = np.loadtxt(alltime_file)  # Load the solution from the file

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
    main(base_folder, algorithms, enemies, enemy_sets, n_hidden, 
         folder_names, num_runs)
