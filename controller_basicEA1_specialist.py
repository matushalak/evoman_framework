#######################################################################################                            #
# Author: Matus Halak       			                                      		  #
# matus.halak@gmail.com
# VISUALISATION SCRIPT     				                              			  #
#######################################################################################

# imports framework
import sys, os

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import argparse

def parse_args():
    '''' Function enabling command-line arguments'''
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Play with chosen best solution of chosen experiment:")
    parser.add_argument('-res', '--results', type=str, required=True, default = False, help= "Provide path to experiments file to analyze:")
    parser.add_argument('-tagainst', '--test_against', type = list, required=True, default = False, help='Provide list of enemies to test against')
    return parser.parse_args()

# get experiment directory
args = parse_args()
exp_dir = args.results

experiment_name = 'controller' + exp_dir
solution = exp_dir + '/best.txt'

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific example
solution_dims = sum(1 for _ in open(solution)) - 1 # (always one extra line)

# get n_hidden neurons from this 
# individual_dims = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
# 5 for 5 output neurons, 26 for 20 inputs, and one bias / weight
n_hidden_neurons = (solution_dims - 5)/26

enems = [int(exp_dir[-1])]
# initializes environment for single objective mode (specialist)  with static enemy and ai player
env = Environment(experiment_name=experiment_name,
                  enemies = enems,
				  playermode="ai",
				  player_controller=player_controller(n_hidden_neurons),
			  	  speed="normal",
				  enemymode="static",
				  level=2,
				  visuals=True)


# tests for each enemy
test_enemies = list(map(int, args.test_against))
for en in test_enemies:

	#Update the enemy
	env.update_parameter('enemies',[en])

	# Load specialist controller
	sol = np.loadtxt(solution)
	print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY '+str(en)+' \n')
	f,pl, el, t = env.play(sol)
      
	print(f, pl, el, t)
	print(f'Gain: {pl-el}')