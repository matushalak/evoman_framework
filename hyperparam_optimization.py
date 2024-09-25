# imports framework
import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
import os
import argparse
from joblib import Parallel, delayed
from pandas import read_csv

#MANUAL INPUT
#num_reps
#n_trials


def objective(trial):
    # Hyperparameter search space
    k = trial.suggest_float('k', 0.1, 10.0)  # Fitness sharing scaling factor
    mutation_rate = trial.suggest_float('mutation_rate', 0.001, 0.1)  # Mutation rate
    sigma = trial.suggest_float('sigma', 0.01, 5.0)  # Mutation sigma
    crossover_rate = trial.suggest_float('crossover_rate', 0.5, 1.0)  # Crossover rate
    alpha = trial.suggest_float('alpha', 0.1, 2.0)  # Recombination factor
    tournament_size = trial.suggest_int('tournament_size', 2, 16)  # Tournament selection size
    elite_fraction = trial.suggest_float('elite_fraction', 0.01, 0.3)  # Elite fraction in survivor selection

    # Replace with your actual evolutionary algorithm call
    # Pass the hyperparameters as arguments
    performance = mean_result_EA1(k, mutation_rate, sigma, crossover_rate, alpha, tournament_size, elite_fraction)
    
    return performance 

def mean_result_EA1(k, mutation_rate, sigma, crossover_rate, alpha, tournament_size, elite_fraction):


    for _ in range(num_reps):
        fitness = basic_ea(PARAMS, enemy, popsize, max_gen, n_hidden, experiment_name,
                 env, run_param_optimization)
        avg_fitness += fitness

    avg_fitness = avg_fitness/num_reps

return avg_fitness




def main():
    '''Main function for basic EA, runs the EA which saves results'''
    # command line arguments for experiment parameters

    experiment_name = 'basic_' + input("Enter Experiment (directory) Name:")
    
    # add enemy name
    experiment_name = experiment_name + f'_{enemy}'
    # directory to save experimental results
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[enemy],
                    playermode="ai",
                    player_controller=player_controller(n_hidden), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)
    
    # default environment fitness is assumed for experiment
    env.state_to_log() # checks environment state


    study = optuna.create_study(direction='maximize')  # If you want to maximize the fitness score
    study.optimize(objective, n_trials=50)  # Perform 50 trials to find the best hyperparameters

    # Print best hyperparameters
    print("Best hyperparameters: ", study.best_params)




#hyperparam--> 
#   for what do we want to optimize? fitness only? Diversity? Mean? 
#   - if we optimize on fitness only, then we should take a few runs and average the results I suppose. 
#   - do we optimize against 1 enemy or against a few and also average those best fitnesses together??
#           ---> maybe do Multi-Objective Optimization??

#what should change
#   - basic_ea should return the best fitness 
#   - we need a different way of saving in basicEA1 --> we don't want to save everything
#       - we need to give it the info hyperparam_optimization

#what to run here:
#   - choose ranges for the hyperparameter
#   - then make a function that runs EA1 multiple times for certain params
#   - this returns the mean fitness for several runs with those parameters
#   - now we do a bayesian search. 

#Then we do: in main
#study = optuna.create_study(direction='maximize')  # If you want to maximize the fitness score
#study.optimize(objective, n_trials=50)  # Perform 50 trials to find the best hyperparameters

# Print best hyperparameters
#print("Best hyperparameters: ", study.best_params)