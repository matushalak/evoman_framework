###############################################################################
# Authors: Matus Halak, Rick Geling, Rens Koppelman, Isis van Loenen
# matus.halak@gmail.com, rickgeling@gmail.com   
###############################################################################

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
import optuna

def parse_args():
    '''' Function enabling command-line arguments'''
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Optimize weights of Controller NN using EA")

    # Define arguments
    parser.add_argument('-name', '--exp_name', type=str, required=False, help="Experiment name")
    parser.add_argument('-mg', '--maxgen', type=int, required=False, default = 100, help="Max generations (eg. 500)")
    parser.add_argument('-nmes', '--enemies', nargs = '+', type = int, required=True, default = False, help='Provide list of enemies to train against')
    parser.add_argument('-mult', '--multi', type=str, required=False, default = 'yes', help="Single or Multienemy")
    parser.add_argument('-trials', '--num_trials', type=int, required=False, default=100, help='Number of bayesian optimization trials') 
    #ADD STORAGE SQL NAME HERE
    
    return parser.parse_args()

def objective(trial, popsize, mg, n_hidden, experiment_name,
                 env, save_gens, num_reps):
    # Hyperparameter search space
    scaling_factor = trial.suggest_float('scaling_factor', 0.01, 0.2)  # Fitness sharing scaling factor
    mutation_rate = trial.suggest_float('mutation_rate', 0.01, 0.75)  # Mutation rate
    sigma_prime = trial.suggest_float('sigma_prime', 0.01, 1.0)  # Mutation sigma
    crossover_rate = trial.suggest_float('crossover_rate', 0.05, 0.75)  # Crossover rate
    alpha = trial.suggest_float('alpha', 0.1, 2.0)  # Recombination factor
    tournament_size = trial.suggest_int('tournament_size', 2, 10)  # Tournament selection size
    elite_fraction = trial.suggest_float('elite_fraction', 0.1, 0.9)  # Elite fraction in survivor selection

    hyperparameters = (scaling_factor, sigma_prime, alpha, tournament_size, elite_fraction, mutation_rate, crossover_rate,
                       popsize, mg)

    # Pass the hyperparameters as arguments
    performance = mean_result_EA1(hyperparameters, n_hidden, experiment_name,
                                    env, save_gens, num_reps)
    
    return performance 

def mean_result_EA1(hyperparameters, n_hidden, experiment_name,
                                    env, save_gens, num_reps):

    avg_fitness = 0
    for i in range(num_reps):
        print(f'Starting repetition number: {i}')
        fitness = basic_ea(hyperparameters, n_hidden, experiment_name,
                    env, save_gens)
        avg_fitness += fitness

    avg_fitness = avg_fitness/num_reps

    return avg_fitness

def main():
    '''Main function for basic EA, runs the EA which saves results'''
    # command line arguments for experiment parameters
    args = parse_args()
    popsize = args.popsize
    mg = args.maxgen
    #cr = args.crossover_rate
    #mr = args.mutation_rate
    n_hidden = args.nhidden
    enemies = args.enemies
    num_trials = args.num_trials

    global multi, fitfunc  
    multi  = 'yes' if args.multi == 'yes' else 'no'
    fitfunc = args.fitness_func
    if fitfunc == 'new':
        print('Using new fitness function')

    save_gens = True
    num_reps = 3

    if isinstance(args.exp_name, str):
        experiment_name = 'basic_' + args.exp_name
    else:
        experiment_name = 'basic_' + input("Enter Experiment (directory) Name:")
    
    # add enemy name
    experiment_name = experiment_name + '_' + f'{str(enemies).strip('[]').replace(',', '').replace(' ', '')}'
    # directory to save experimental results
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=enemies,
                    multiplemode=multi, 
                    playermode="ai",
                    player_controller=player_controller(n_hidden), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)
    
    # default environment fitness is assumed for experiment
    env.state_to_log() # checks environment state

    study = optuna.create_study(direction='maximize',
                                        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
                                        study_name="08_10_nightrun")  # If you want to maximize the fitness score
    study.optimize(lambda trial: objective(trial, popsize, mg, n_hidden, experiment_name,
                     env, save_gens, num_reps), n_trials=num_trials)

    # Print best hyperparameters
    print("Best hyperparameters: ", study.best_params)



if __name__ == '__main__':
    main()