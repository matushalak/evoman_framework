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
    parser.add_argument('-res', '--results', type=str, required=False, default = False, help= "Provide path to experiments file to analyze:")
    parser.add_argument('-tagainst', '--test_against', nargs = '+', type = int, required=True, default = False, help='Provide list of enemies to test against')
    return parser.parse_args()

all_best = False
# get experiment directory
args = parse_args()
if isinstance(args.results, str):
	exp_dir = args.results
	if 'alltime.txt' in os.listdir(exp_dir):
		solution = exp_dir + '/alltime.txt'
	else:
		solution = exp_dir + '/best.txt'
else:
    exp_dir = 'basic_solutions'
    solutions = os.listdir(exp_dir)
    all_best = True
    solution = exp_dir + '/' + solutions[0]

experiment_name = exp_dir

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

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


# tests for each enemy
test_enemies = args.test_against
for en in test_enemies:
	trained_against = exp_dir[-1] if isinstance(args.results, str) else en

	#Update the enemy
	env.update_parameter('enemies',[en])

	# Load specialist controller
	if all_best == True:
		sol_i = solutions.index(f'{en}best.txt')
		sol = np.loadtxt(exp_dir+'/'+solutions[sol_i])
	else:
		sol = np.loadtxt(solution)
	print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY ' + str(trained_against) + '\n')
	f,pl, el, t = env.play(sol)
      
	print(f, pl, el, t)
	print(f'Gain: {pl-el}')