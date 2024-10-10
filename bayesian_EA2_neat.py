###############
# @matushalak
# code adapted from NEAT-python package, XOr & OpenAI-Lander examples
# eg. https://github.com/CodeReclaimers/neat-python/blob/master/examples/xor/evolve-feedforward-parallel.py
###############
import multiprocessing
import os
import neat
from evoman.environment import Environment
from demo_controller import player_controller
from bayesian_neat_controller import bayesian_neat_controller
from time import time
from pandas import DataFrame
import argparse
import pickle as pkl

import optuna
import random
import functools

########
# NOTE: change the ranges of the hyperparameter search space below in the function 'objective' (this could be made more robust ofc). 
########


# --------------------------------------------------------NOTE, BELOW THIS: NEAT ALGORTIHM 
# - difference to EA2_neat: moved env = Environment to eval_genome() function (has to be there, else unknown!)
#                           made run() return the final best fitness
#                           added num_reps and num_trials, l_and_c and dbname in parser. Set a default exp_name (to easily continue a study)
#                           added some variables to global. 
#                           slightly changed the way of processing 'name'

def parse_args():
    '''' Function enabling command-line arguments'''
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Optimize weights of Controller NN using EA")

    # Define arguments
    parser.add_argument('-name', '--exp_name', type=str, required=False, default='bayes_run', help="Experiment name")
    parser.add_argument('-dbname', '--db_name', type=str, required=False, default='bayesian_neat_TEST', help="Database name")
    parser.add_argument('-mg', '--maxgen', type=int, required=False, default = 5, help="Max generations (eg. 500)")
    parser.add_argument('-nmes', '--enemies', nargs = '+', type = int, required=False, default = [5, 6], help='Provide list of enemies to train against')
    parser.add_argument('-mult', '--multi', type=str, required=False, default = 'yes', help="Single or Multienemy")
    parser.add_argument('-trials', '--num_trials', type=int, required=False, default=3, help='Number of bayesian optimization trials') 
    parser.add_argument('-reps', '--num_reps', type=int, required=False, default=2, 
                        help='Number of NEAT repititions with the same set of params') 
    parser.add_argument('-lc', '--l_and_c', type=bool, required=False, default = True, 
                        help="Loads and continues previous study if exists and set to True")

    return parser.parse_args()


args = parse_args()
global env, maxgen, name, enemies, num_reps, num_trials, load_and_continue, db_name, multi
maxgen = args.maxgen
name = args.exp_name  #NOTE: option 1: change the name if you do not want to continue a study
enemies = args.enemies
num_reps = args.num_reps
num_trials = args.num_trials
load_and_continue = args.l_and_c #NOTE: option 2: set to False and overwrite what was done before (in case same name and db_name)
db_name = args.db_name #NOTE: option 3: change the database name to get a complete new database of studies
multi  = 'yes' if args.multi == 'yes' else 'no'

# add enemy names
name = name + '_' + f'EN{str(enemies).strip('[]').replace(',', '').replace(' ', '')}' + f'reps{num_reps}' + f'mg{maxgen}'
if not os.path.exists(name):
    os.makedirs(name)

#NOTE: env used to be here

os.environ["SDL_VIDEODRIVER"] = "dummy"

def eval_genome(genome, config):
    '''
    Parallelized version
    '''
    env = Environment(experiment_name=name,
                enemies=enemies,
                multiplemode=multi, 
                playermode="ai",
                player_controller=bayesian_neat_controller(config), # you  can insert your own controller here
                enemymode="static",
                level=2,
                speed="fastest",
                visuals=False)
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness ,p,e,t = env.play(pcont=net)
    return fitness

def save_stats(StatsReporter):
    results = DataFrame({'gen':list(range(maxgen)),
                         'best':StatsReporter.get_fitness_stat(max),
                         'mean':StatsReporter.get_fitness_mean(),
                         'sd':StatsReporter.get_fitness_stdev(),
                         'med':StatsReporter.get_fitness_median(),
                         'worst':StatsReporter.get_fitness_stat(min)})
    results.to_csv(name + '/results.txt')

def run(config_path):
    start = time()

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
							neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    
    # population
    pop = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Run for N generations
    # parallel
    parallel_evals = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(parallel_evals.evaluate, maxgen)

    # winner = pop.run(eval_genomes, 50) # classic
    winn_gene = stats.best_genome()
    winner_net = neat.nn.FeedForwardNetwork.create(winn_gene, config)

    # save results
    save_stats(stats)
    # save controller
    with open(name + '/best.pkl', 'wb') as f:
        pkl.dump(winner_net, f)

    end = time()-start
    print(end)

    # Display winning genome
    print('\nBest genome:\n{!s}'.format(winner))

    return winner.fitness


# -------------------------------------------------------- NOTE, BELOW THIS: BAYESIAN OPTIMIZATION

def mean_result_NEAT(config_path, num_reps):

    #TODO check if this can be executed parallel.. Or if it causes nested paralellism problems..
    print(f'Start running a new set of parameters..')
    avg_fitness = 0
    for i in range(num_reps):
        print(f'Starting repetition number: {i+1} out of {num_reps}')
        fitness = run(config_path)   
        avg_fitness += fitness

    avg_fitness = avg_fitness/num_reps

    return avg_fitness


def make_config(p_add_connection:float, 
                p_remove_connection:float, 
                p_add_node:float,
                p_remove_node:float,
                N_starting_hidden_neurons:int):

    os.makedirs("configs", exist_ok=True)
    # how many config files do we have already
    n_configs = sum([1 if 'neat_config' in f else 0 for f in os.listdir('configs')])    
    
    # this will be the Nth config file
    file_name = f'configs/neat_config{n_configs+1}.txt'

    keywords = {'conn_add_prob':p_add_connection,
                'conn_delete_prob':p_remove_connection,
                'node_add_prob':p_add_node,
                'node_delete_prob':p_remove_node,
                'num_hidden':N_starting_hidden_neurons}

    # open default config
    with open('neat_config.txt', 'r') as default:
        with open(file_name, 'w') as new:
            for line in default:
                if any(k in line for k in keywords) == True:
                    parameter = next(k for k in keywords if k in line)
                    start, end = line.split('=')
                    # update with our parameter value
                    end = str(keywords[parameter])
                    new_line = start + '= ' + end +'\n'
                    new.write(new_line)
                # copy all other lines
                else:
                    new.write(line)
    # which config to load
    print(f'config to lead: {file_name}')

    return file_name


def objective(trial, num_reps):
    
    # set the ranges for the parameters to optimize and build the trial
    #TODO: if we change the params and ranges, do we need to change the study??
    p_add_connection = trial.suggest_float('p_add_connection', 0.4, 0.6)  
    p_remove_connection = trial.suggest_float('p_remove_connection', 0.4, 0.6)  
    p_add_node = trial.suggest_float('p_add_node', 0.1, 0.3)  
    p_remove_node = trial.suggest_float('p_remove_node', 0.1, 0.3)  
    N_starting_hidden_neurons = trial.suggest_int('N_starting_hidden_neurons', 8, 12)  

    # make a config file with the new chosen parameters
    config_path = make_config(p_add_connection, 
                    p_remove_connection, 
                    p_add_node,
                    p_remove_node,
                    N_starting_hidden_neurons)

    # pass the config file to mean results which runs the NEAT
    performance = mean_result_NEAT(config_path, num_reps)
    
    return performance 


def bayesian_optimization():
    # no arguments taken since we use global variables

    # create study. NOTE: sampling algorithm is set to TPE (can also be set to NSGAII, QMC, GP, etc.)
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(),    
                                direction='maximize',
                                storage=f'sqlite:///{db_name}.db',  # Specify the storage URL here.
                                study_name=name,
                                load_if_exists=load_and_continue) 
    study.optimize(lambda trial: objective(trial, num_reps), n_trials=num_trials)

    # Print best hyperparameters
    print("Best hyperparameters: ", study.best_params)


if __name__ == '__main__':
    bayesian_optimization() 