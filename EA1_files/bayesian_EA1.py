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

from EA1_optimizer import ClassicEA  #CHANGED

def parse_args():
    '''' Function enabling command-line arguments'''
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Optimize weights of Controller NN using EA")

    # Define arguments
    parser.add_argument('-name', '--exp_name', type=str, required=False, default='bayes_run', help="Experiment name")
    parser.add_argument('-dbname', '--db_name', type=str, required=False, default='bayesian_EA1_TEST', help="Database name")
    parser.add_argument('-pop', '--popsize', type=int, required=False, default = 150, help="Population size (eg. 100)")
    parser.add_argument('-mg', '--maxgen', type=int, required=False, default = 100, help="Max generations (eg. 500)")
    parser.add_argument('-nh', '--nhidden', type=int, required=False, default = 10, help="Number of Hidden Neurons (eg. 10)")
    parser.add_argument('-nmes', '--enemies', nargs = '+', type = int, required=True, default = False, help='Provide list of enemies to train against')
    parser.add_argument('-mult', '--multi', type=str, required=False, default = 'yes', help="Single or Multienemy")
    parser.add_argument('-fit', '--fitness_func', type=str, required=False, default='old', help = 'Which Fitness function to use? [old / new]')
    parser.add_argument('-trials', '--num_trials', type=int, required=False, default=100, help='Number of bayesian optimization trials') 
    parser.add_argument('-reps', '--num_reps', type=int, required=False, default=3, 
                        help='Number of NEAT repititions with the same set of params') 
    parser.add_argument('-lc', '--l_and_c', type=bool, required=False, default = True, 
                        help="Loads and continues previous study if exists and set to True")


    return parser.parse_args()

def objective(trial, popsize, mg, n_hidden, experiment_name,
                 env, save_gens, num_reps):
    # Hyperparameter search space
    scaling_factor = trial.suggest_float('scaling_factor', 0.01, 0.25)  # Fitness sharing scaling factor: max_dist usually 15, I think going higher than 0.25 is strange
    mutation_rate = trial.suggest_float('mutation_rate', 0.01, 0.75)  # Mutation rate
    sigma_prime = trial.suggest_float('sigma_prime', 0.01, 0.5)  # Mutation sigma: I think higher than 0.5 results in a lot of clipping
    crossover_rate = trial.suggest_float('crossover_rate', 0.1, 0.9)  # Crossover rate
    alpha = trial.suggest_float('alpha', 0.1, 1.0)  # Recombination factor: they say 0.5 should be optimal, I think lower is better
    tournament_size = trial.suggest_int('tournament_size', 2, 15)  # Tournament selection size: from examples 7/8 seems good
    elite_fraction = trial.suggest_float('elite_fraction', 0.05, 0.5)  # Elite fraction in survivor selection: no more than 50% elitsm right?

    hyperparameters = {
        "scaling_factor": scaling_factor,
        "sigma_prime": sigma_prime,
        "alpha": alpha,
        "tournament_size": tournament_size,
        "elite_fraction": elite_fraction,
        "mutation_rate": mutation_rate,
        "crossover_rate": crossover_rate,
        "popsize": popsize,
        "max_gen": mg
    }

    # Pass the hyperparameters as arguments
    performance = mean_result_EA1(hyperparameters, n_hidden, experiment_name,
                                    env, save_gens, num_reps)
    
    return performance 

def mean_result_EA1(hyperparameters, n_hidden, experiment_name,
                                    env, save_gens, num_reps):

    avg_fitness = 0
    for i in range(num_reps):
        print(f'Starting repetition number: {i+1} out of {num_reps}')
        ea = ClassicEA(hyperparameters, n_hidden, experiment_name, env)  #CHANGED
        fitness = ea.run_evolution()  #CHANGED
        avg_fitness += fitness

    avg_fitness = avg_fitness/num_reps

    return avg_fitness

def main():
    '''Main function for basic EA, runs the EA which saves results'''
    
    # command line arguments for experiment parameters
    args = parse_args()

    popsize = args.popsize
    mg = args.maxgen
    n_hidden = args.nhidden
    enemies = args.enemies
    num_reps = args.num_reps
    num_trials = args.num_trials
    load_and_continue = args.l_and_c #NOTE: option 2: set to False and overwrite what was done before (in case same name and db_name)
    db_name = args.db_name #NOTE: option 3: change the database name to get a complete new database of studies

    global multi, fitfunc  
    multi  = 'yes' if args.multi == 'yes' else 'no'
    fitfunc = args.fitness_func
    if fitfunc == 'new':
        print('Using new fitness function')

    save_gens = True
    
    # add enemy name
    experiment_name = args.exp_name + '_' + f'EN{str(enemies).strip('[]').replace(',', '').replace(' ', '')}' + '_' + f'reps{num_reps}' + '_' + f'mg{mg}'
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

    # create study. NOTE: sampling algorithm is set to TPE (can also be set to NSGAII, QMC, GP, etc.)
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(),    
                                direction='maximize',
                                storage=f'sqlite:///{db_name}.db',  # Specify the storage URL here.
                                study_name=experiment_name,
                                load_if_exists=load_and_continue) 
    study.optimize(lambda trial: objective(trial, popsize, mg, n_hidden, experiment_name,
                     env, save_gens, num_reps), n_trials=num_trials)

    # Print best hyperparameters
    print("Best hyperparameters: ", study.best_params)


if __name__ == '__main__':
    main()