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

from EA1_optimizer import ClassicEA  #CHANGED


def parse_args():  # Unchanged
    '''' Function enabling command-line arguments'''  # Unchanged
    # Initialize the argument parser  # Unchanged
    parser = argparse.ArgumentParser(description="Optimize weights of Controller NN using EA")  # Unchanged

    # Define arguments  # Unchanged
    parser.add_argument('-name', '--exp_name', type=str, required=False, help="Experiment name")  # Unchanged
    parser.add_argument('-pop', '--popsize', type=int, required=False, default=150, help="Population size (eg. 100)")  # Unchanged
    parser.add_argument('-mg', '--maxgen', type=int, required=False, default=100, help="Max generations (eg. 500)")  # Unchanged
    parser.add_argument('-cr', '--crossover_rate', type=float, required=False, default=0.85, help="Crossover rate (e.g., 0.8)")  # Unchanged
    parser.add_argument('-mr', '--mutation_rate', type=float, required=False, default=0.25, help="Mutation rate (e.g., 0.05)")  # Unchanged
    parser.add_argument('-nh', '--nhidden', type=int, required=False, default=10, help="Number of Hidden Neurons (eg. 10)")  # Unchanged
    parser.add_argument('-tst', '--test', type=bool, required=False, default=False, help="Train or Test (default = Train)")  # Unchanged
    parser.add_argument('-nmes', '--enemies', nargs='+', type=int, required=False, default=[6, 7 ,4], help='Provide list of enemies to train against')  # Unchanged
    parser.add_argument('-mult', '--multi', type=str, required=False, default='yes', help="Single or Multienemy")  # Unchanged
    parser.add_argument('-fit', '--fitness_func', type=str, required=False, default='old', help='Which Fitness function to use? [old / new]')  # Unchanged

    return parser.parse_args()  # Unchanged


def main():
    '''Main function for Classic EA, runs the EA which saves results'''  #CHANGED

    # command line arguments for experiment parameters
    args = parse_args()  # Unchanged
    popsize = args.popsize  # Unchanged
    mg = args.maxgen  # Unchanged
    cr = args.crossover_rate  # Unchanged
    mr = args.mutation_rate  # Unchanged
    n_hidden = args.nhidden  # Unchanged
    enemies = args.enemies  # Unchanged
    global multi, fitfunc  # Unchanged
    multi = 'yes' if args.multi == 'yes' else 'no'  # Unchanged
    fitfunc = args.fitness_func  # Unchanged

    hyperparameters = {
    "scaling_factor": 0.08,
    "sigma_prime": 0.1,
    "alpha": 0.3,
    "tournament_size": 8,
    "elite_fraction": 0.1,
    "mutation_rate": mr,
    "crossover_rate": cr,
    "popsize": popsize,
    "max_gen": mg
    }

    if fitfunc == 'new':  # Unchanged
        print('Using new fitness function')  # Unchanged

    if isinstance(args.exp_name, str):  # Unchanged
        experiment_name = 'classic_' + args.exp_name  #CHANGED
    else:  # Unchanged
        experiment_name = 'classic_' + input("Enter Experiment (directory) Name:")  #CHANGED

    # add enemy name
    experiment_name = experiment_name + '_' + f'{str(enemies).strip("[]").replace(",", "").replace(" ", "")}'  # Unchanged

    # directory to save experimental results
    if not os.path.exists(experiment_name):  # Unchanged
        os.makedirs(experiment_name)  # Unchanged

    # choose this for not using visuals and thus making experiments faster
    headless = True  # Unchanged
    if headless:  # Unchanged
        os.environ["SDL_VIDEODRIVER"] = "dummy"  # Unchanged

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,  # Unchanged
                      enemies=enemies,  # Unchanged
                      multiplemode=multi,  # Unchanged
                      playermode="ai",  # Unchanged
                      player_controller=player_controller(n_hidden),  # you can insert your own controller here # Unchanged
                      enemymode="static",  # Unchanged
                      level=2,  # Unchanged
                      speed="fastest",  # Unchanged
                      visuals=False)  # Unchanged

    # default environment fitness is assumed for experiment
    env.state_to_log()  # Unchanged

    if args.test == True:
        best_solution = np.loadtxt(experiment_name+'/best.txt')
        print( '\n RUNNING SAVED BEST SOLUTION \n')
        env.update_parameter('speed','normal')
        vfitness, vplayerlife, venemylife, vtime = run_game(env, best_solution, test = True)
        print('vfitness, vplayerlife, venemylife, vtime:\n',
              vfitness, vplayerlife, venemylife, vtime)
        sys.exit(0)

    if not os.path.exists(experiment_name + '/evoman_solstate'):  # Unchanged
        print('\nNEW EVOLUTION\n')  # Unchanged
        # Create an instance of the ClassicEA class and run evolution  #CHANGED
        ea = ClassicEA(hyperparameters, n_hidden, experiment_name, env)  #CHANGED
        final_fitness = ea.run_evolution()  #CHANGED
    else:  # Unchanged
        # Continue existing evolution using ClassicEA  #CHANGED
        ea = ClassicEA(hyperparameters, n_hidden, experiment_name, env)  #CHANGED
        final_fitness = ea.run_evolution()  #CHANGED


# # runs game (evaluate fitness for 1 individual)
def run_game(env:Environment,individual, test=False):
    '''Runs game and returns individual solution's fitness'''
    # vfitness, vplayerlife, venemylife, vtime
    if isinstance(individual, float):
        breakpoint()
    fitness ,p,e,t = env.play(pcont=individual)
    if test == False:
        if fitfunc == 'new':
            return (p-(2*e)) - 0.01*t
        else: 
            return fitness
    else:
        return fitness ,p,e,t


if __name__ == '__main__':  # Unchanged
    main()  # Unchanged
