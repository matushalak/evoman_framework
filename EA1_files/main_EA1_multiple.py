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

from EA1_optimizer import ClassicEA  # ADDED

def parse_args():
    '''' Function enabling command-line arguments'''
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Optimize weights of Controller NN using EA")

    # Define arguments
    parser.add_argument('-name', '--exp_name', type=str, required=False, help="Experiment name")  # Unchanged
    parser.add_argument('-pop', '--popsize', type=int, required=False, default=150, help="Population size (eg. 100)")  # Unchanged
    parser.add_argument('-mg', '--maxgen', type=int, required=False, default=5, help="Max generations (eg. 500)")  # Unchanged
    parser.add_argument('-cr', '--crossover_rate', type=float, required=False, default=0.85, help="Crossover rate (e.g., 0.8)")  # Unchanged
    parser.add_argument('-mr', '--mutation_rate', type=float, required=False, default=0.25, help="Mutation rate (e.g., 0.05)")  # Unchanged
    parser.add_argument('-nh', '--nhidden', type=int, required=False, default=10, help="Number of Hidden Neurons (eg. 10)")  # Unchanged
    parser.add_argument('-tst', '--test', type=bool, required=False, default=False, help="Train or Test (default = Train)")  # Unchanged
    parser.add_argument('-mult', '--multi', type=str, required=False, default='yes', help="Single or Multienemy")  # Unchanged
    parser.add_argument('-fit', '--fitness_func', type=str, required=False, default='old', help='Which Fitness function to use? [old / new]')  # Unchanged
    parser.add_argument('-nmes', '--enemies', nargs='+', type=int, required=False, default=[[5, 6], [2,3]], help='Provide list(s) of enemies to train against')  # Unchanged
    #EA1_line_plot_runs     
    parser.add_argument('-dir', '--directory', type=str, default='NEW_TEST_EA1_line_plot_runs', required=False, help="Directory to save runs")
    parser.add_argument('-nruns', '--num_runs', type=int, required=False, default=100, help="Number of repetitive ClassicEA runs")  # Unchanged

    return parser.parse_args()


def main():
    '''Main function for basic EA, runs the EA for multiple enemies'''
    
    # command line arguments for experiment parameters
    args = parse_args()  # Unchanged
    popsize = args.popsize  # Unchanged
    mg = args.maxgen  # Unchanged
    cr = args.crossover_rate  # Unchanged
    mr = args.mutation_rate  # Unchanged
    n_hidden = args.nhidden  # Unchanged
    enemy_sets = args.enemies  # Unchanged
    global multi, fitfunc  # Unchanged
    multi = 'yes' if args.multi == 'yes' else 'no'  # Unchanged
    fitfunc = args.fitness_func  # Unchanged
    base_dir = args.directory
    num_runs = args.num_runs

    hyperparameters = {
    "scaling_factor": 0.15,
    # "sigma_prime": 0.05,
    "alpha": 0.5,
    "tournament_size": 6,
    "elite_fraction": 0.8,
    "mutation_rate": mr,
    "crossover_rate": cr,
    "popsize": popsize,
    "max_gen": mg,
    'specialist_frequency':20
    }

    # Create the base directory if it does not exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # Run the EA for each enemy
    for enemies in enemy_sets:
        enemy_dir = os.path.join(base_dir, f'EN{enemies}')
        if not os.path.exists(enemy_dir):
            os.makedirs(enemy_dir)

        for run in range(1, num_runs+1):  # Run the EA e.g. 10 times for each enemy
            run_dir = os.path.join(enemy_dir, f'run_{run}_EN{enemies}')
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)

            env = Environment(experiment_name=run_dir,
                        enemies=enemies,
                        multiplemode=multi,  # Unchanged
                        playermode="ai",
                        player_controller=player_controller(n_hidden),
                        enemymode="static",
                        level=2,
                        speed="fastest",
                        visuals=False)
            
            # default environment fitness is assumed for experiment
            env.state_to_log() # checks environment state

            # Run the ClassicEA for this enemy and run number
            print(f'\nRunning EA1 (classis) for enemy (set) {enemies}, run {run}\n')
            ea = ClassicEA(hyperparameters, n_hidden, run_dir, env)  # ADDED
            final_fitness = ea.run_evolution()  # ADDED


if __name__ == '__main__':
    main()