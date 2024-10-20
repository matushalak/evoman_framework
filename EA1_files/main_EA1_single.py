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

from EA1_optimizer import ClassicEA  


def parse_args():  
    '''' Function enabling command-line arguments'''  
    # Initialize the argument parser  
    parser = argparse.ArgumentParser(description="Optimize weights of Controller NN using EA")  # 

    # Define arguments  # 
    parser.add_argument('-name', '--exp_name', type=str, required=False, help="Experiment name")  # 
    parser.add_argument('-pop', '--popsize', type=int, required=False, default=150, help="Population size (eg. 100)")  
    parser.add_argument('-mg', '--maxgen', type=int, required=False, default=100, help="Max generations (eg. 500)")  
    parser.add_argument('-nh', '--nhidden', type=int, required=False, default=10, help="Number of Hidden Neurons (eg. 10)")  
    parser.add_argument('-tst', '--test', type=bool, required=False, default=False, help="Train or Test (default = Train)")  
    parser.add_argument('-nmes', '--enemies', nargs='+', type=int, required=False, default=[6, 7 ,4], help='Provide list of enemies to train against')  
    parser.add_argument('-mult', '--multi', type=str, required=False, default='yes', help="Single or Multienemy")  
    parser.add_argument('-fit', '--fitness_func', type=str, required=False, default='old', help='Which Fitness function to use? [old / new]')  

    return parser.parse_args()  


def main():
    '''Main function for Classic EA, runs the EA which saves results'''  

    # command line arguments for experiment parameters
    args = parse_args()  
    popsize = args.popsize  
    mg = args.maxgen  
    n_hidden = args.nhidden  
    enemies = args.enemies  
    global multi, fitfunc  
    multi = 'yes' if args.multi == 'yes' else 'no'  
    fitfunc = args.fitness_func  

    hyperparameters = {
    "scaling_factor": 0.236170145228546,
    # "sigma_prime": 0.321165796694383,
    "alpha": 0.337159209115772,
    "tournament_size": 4,
    "elite_fraction": 0.361110061941829,
    "mutation_rate": 0.500170314290922,
    "crossover_rate": 0.290183483321576,
    "popsize": popsize,
    "max_gen": mg,
    'specialist_frequency' : 20
    }

    if fitfunc == 'new':  
        print('Using new fitness function')  

    if isinstance(args.exp_name, str):  
        experiment_name = 'classic_' + args.exp_name  
    else:  
        experiment_name = 'classic_' + input("Enter Experiment (directory) Name:")  

    # add enemy name
    experiment_name = experiment_name + '_' + f'{str(enemies).strip("[]").replace(",", "").replace(" ", "")}'  

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
                      player_controller=player_controller(n_hidden),  # you can insert your own controller here 
                      enemymode="static",  
                      level=2,  
                      speed="fastest",  
                      visuals=False)  

    # default environment fitness is assumed for experiment
    env.state_to_log()  

    if args.test == True:
        best_solution = np.loadtxt(experiment_name+'/best.txt')
        print( '\n RUNNING SAVED BEST SOLUTION \n')
        env.update_parameter('speed','normal')
        vfitness, vplayerlife, venemylife, vtime = run_game(env, best_solution, test = True)
        print('vfitness, vplayerlife, venemylife, vtime:\n',
              vfitness, vplayerlife, venemylife, vtime)
        sys.exit(0)

    if not os.path.exists(experiment_name + '/evoman_solstate'):  
        print('\nNEW EVOLUTION\n')  
        # Create an instance of the ClassicEA class and run evolution  
        ea = ClassicEA(hyperparameters, n_hidden, experiment_name, env)  
        final_fitness = ea.run_evolution()  
    else:  
        # Continue existing evolution using ClassicEA  
        ea = ClassicEA(hyperparameters, n_hidden, experiment_name, env)  
        final_fitness = ea.run_evolution()  


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
            return p - e
    else:
        return fitness ,p,e,t


if __name__ == '__main__':  
    main()  
