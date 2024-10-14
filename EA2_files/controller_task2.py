#######################################################################################                            
###############################################################################
# Authors: Matus Halak, Rick Geling, Rens Koppelman, Isis van Loenen
# matus.halak@gmail.com, rickgeling@gmail.com   
# VISUALISATION SCRIPT     				                              			  
#######################################################################################

# imports framework
import os

from evoman.environment import Environment
# from demo_controller import player_controller
from neat_controller import neat_controller

# imports other libs
import numpy as np
import argparse
import pickle as pkl

def parse_args():
    '''' Function enabling command-line arguments'''
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Play with chosen best solution of chosen experiment:")
    parser.add_argument('-res', '--results', type=str, required=False, default = False, help= "Provide path to experiments file to analyze:")
    parser.add_argument('-tagainst', '--test_against', nargs = '+', type = int, required=False, default = 10, help='Provide list of enemies to test against')
    return parser.parse_args()

# get experiment directory
args = parse_args()
if isinstance(args.results, str):
	exp_dir = args.results
	if 'alltime.txt' in os.listdir(exp_dir):
		solution = exp_dir + '/alltime.txt'
	else:
		if 'neat' not in exp_dir:
			solution = exp_dir + '/best.txt'
		else:
			solution = exp_dir + '/best.pkl'

experiment_name = exp_dir

# Basic EA
if 'neat' not in experiment_name:
	# Update the number of neurons for this specific example
	solution_dims = sum(1 for _ in open(solution)) - 1 # (always one extra line)

	# get n_hidden neurons from this 
	# individual_dims = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
	# 5 for 5 output neurons, 26 for 20 inputs, and one bias / weight
	n_hidden_neurons = int(round((solution_dims - 5)/26))

	# initializes environment for single objective mode (specialist)  with static enemy and ai player
	env = Environment(experiment_name=experiment_name,
					playermode="ai",
					player_controller=player_controller(n_hidden_neurons),
					speed="normal",
					enemymode="static",
					level=2,
					visuals=True)
	# Load controller
	sol = np.loadtxt(solution)

# NEAT
else:
	env = Environment(experiment_name=experiment_name,
					playermode="ai",
					player_controller=neat_controller(config_path='neat_config.txt'),
					speed="normal",
					enemymode="static",
					level=2,
					visuals=True)
	
	# Load controller
	with open(solution, 'rb') as f:
		sol = pkl.load(f)

# tests for each enemy
test_enemies = args.test_against
if test_enemies == 10:
	test_enemies = list(range(1,9))
for en in test_enemies:
	trained_against = exp_dir[-1] if isinstance(args.results, str) else en

	#Update the enemy
	env.update_parameter('enemies',[en])
	print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY ' + str(trained_against) + '\n')
	f,pl, el, t = env.play(pcont = sol)
      
	print(f, pl, el, t)
	print(f'Gain: {pl-el}')