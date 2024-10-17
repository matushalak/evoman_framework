###############################################################################
# Authors: Matus Halak, Rick Geling, Rens Koppelman, Isis van Loenen
# matus.halak@gmail.com, rickgeling@gmail.com   
###############################################################################

import sys

from EA1_files.evoman.environment import Environment
from EA1_files.demo_controller import player_controller
import os
import json
import numpy as np

# Set the base folder for dummy data
base_folder = 'gain_results'
os.makedirs(base_folder, exist_ok=True)
algorithms = ['EA1']#, 'EA2']
folder_names = {
    'EA1': 'EA1_final_runs',  # Folder for EA1
    #'EA2': 'EA2_report_results'  # Folder for EA2
}
enemies = [1, 2, 3, 4, 5, 6, 7, 8]
n_hidden = 10


# Function to save dummy data in JSON format
def save_json_data(algo_folder, enemy, data):
    os.makedirs(algo_folder, exist_ok=True)
    file_path = os.path.join(algo_folder, f'enemy_{enemy}_gains.json')
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved: {file_path}")


# # runs game (evaluate fitness for 1 individual)
def run_game(env:Environment, individual):
    '''Runs game and returns individual solution's fitness'''
    
    # vfitness, vplayerlife, venemylife, vtime
    fitness,p,e,t = env.play(pcont=individual)

    return fitness,p,e,t

def main(base_folder,algorithms,enemies,n_hidden,folder_names):

    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=base_folder,
                    playermode="ai",
                    player_controller=player_controller(n_hidden), # you can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)

    # Go through the algorithms:
    for algo in algorithms:
        for enemy in enemies:
            #dictionary to store gains for the current algorithm and enemy
            gains_dict = {}

            #Update the enemy
            env.update_parameter('enemies',[enemy])

            # Iterate through the runs (1 to 10)
            for run in range(1, 11):
                folder_name = folder_names[algo]
                pass_folder = os.path.join('EA1_files', folder_name)
                run_folder = os.path.join(pass_folder, f'EN{enemy}', f'run_{run}_EN{enemy}')
                alltime_file = os.path.join(run_folder, 'alltime.txt')
                
                #check if the alltime.txt file exists
                if os.path.exists(alltime_file):
                    sol = np.loadtxt(alltime_file)  # Load the solution from the file
                    gains = []

                    #play the game 5 times for each run
                    for j in range(9):
                        enemy_to_play = j+1
                        env = Environment(experiment_name=run_folder,  # Unchanged
                                        enemies=enemy_to_play,  # Unchanged
                                        playermode="ai",  # Unchanged
                                        player_controller=player_controller(n_hidden),  # you can insert your own controller here # Unchanged
                                        enemymode="static",  # Unchanged
                                        level=2,  # Unchanged
                                        speed="fastest",  # Unchanged
                                        visuals=False)  # Unchanged

                        fitness, p, e, t = run_game(env, sol)
                        gain = p - e  # Calculate gain as p (player life) - e (enemy life)
                        gains.append(gain)

                    #store the 5 gains for this run in the dictionary
                    gains_dict[f'run_{run}'] = gains

                else:
                    print(f"File not found: {alltime_file}")

            #save the gains for the current algorithm and enemy
            save_json_data(os.path.join(base_folder, algo), enemy, gains_dict)


if __name__ == "__main__":
    main(base_folder, algorithms, enemies, n_hidden, folder_names)
