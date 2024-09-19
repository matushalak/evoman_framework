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

experiment_name = 'basic/controller_specialist_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific example
n_hidden_neurons = 10

enems = [2]
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
for en in range(6,9):

	#Update the enemy
	env.update_parameter('enemies',[en])

	# Load specialist controller
	sol = np.loadtxt('basic_parallel/best.txt')
	print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY '+str(en)+' \n')
	f,pl, el, t = env.play(sol)
      
	print(f, pl, el, t)