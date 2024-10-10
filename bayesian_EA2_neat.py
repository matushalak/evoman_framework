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
from neat_controller import neat_controller
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
# - difference to EA2_neat: moved env = Environment to run() function
#                           made run() return the final best fitness
#                           added num_reps and num_trials in parser and global environment

def parse_args():
    '''' Function enabling command-line arguments'''
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Optimize weights of Controller NN using EA")

    # Define arguments
    parser.add_argument('-name', '--exp_name', type=str, required=False, help="Experiment name")
    parser.add_argument('-mg', '--maxgen', type=int, required=False, default = 10, help="Max generations (eg. 500)")
    parser.add_argument('-nmes', '--enemies', nargs = '+', type = int, required=False, default = [5, 6], help='Provide list of enemies to train against')
    parser.add_argument('-mult', '--multi', type=str, required=False, default = 'yes', help="Single or Multienemy")
    parser.add_argument('-trials', '--num_trials', type=int, required=False, default=3, help='Number of bayesian optimization trials') 
    parser.add_argument('-reps', '--num_reps', type=int, required=False, default=2, help='Number of NEAT repititions with the same set of params') 

    return parser.parse_args()

args = parse_args()
global env, cfg, name, enemies, multi, maxgen
cfg = 'neat_config.txt'
maxgen = args.maxgen
enemies = args.enemies
num_reps = args.num_reps
num_trials = args.num_trials
multi  = 'yes' if args.multi == 'yes' else 'no'

if isinstance(args.exp_name, str):
    name = 'neat_' + args.exp_name
else:
    name = 'neat_' + input("Enter Experiment (directory) Name:")

# add enemy names
name = name + '_' + f'{str(enemies).strip('[]').replace(',', '').replace(' ', '')}'
if not os.path.exists(name):
    os.makedirs(name)

#NOTE: here env used to be
# env = Environment(experiment_name=name,
#             enemies=enemies,
#             multiplemode=multi, 
#             playermode="ai",
#             player_controller=neat_controller(config_path=cfg), # you  can insert your own controller here
#             enemymode="static",
#             level=2,
#             speed="fastest",
#             visuals=False)

os.environ["SDL_VIDEODRIVER"] = "dummy"

def eval_genome(genome, config):
    '''
    Parallelized version
    '''
    
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

# def print_env():
#     print(f'NEW RUN, NEW ENV')
#     print(env)

def run(config_path):

    global env

    #TODO ENV HERE? 

    env = Environment(experiment_name=name,
                enemies=enemies,
                multiplemode=multi, 
                playermode="ai",
                player_controller=neat_controller(config_path=config_path), # you  can insert your own controller here
                enemymode="static",
                level=2,
                speed="fastest",
                visuals=False)

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

    return random.uniform(80, 95)


# -------------------------------------------------------- NOTE, BELOW THIS: BAYESIAN OPTIMIZATION

def mean_result_NEAT(config_path, num_reps):

    #TODO check if this can be runned parallel... 

    avg_fitness = 0
    for i in range(num_reps):
        print(f'Starting repetition number: {i}')
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


def bayesian_optimization(num_trials, num_reps):
    
    study = optuna.create_study(direction='maximize',
                                        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
                                        study_name=name)  #TODO see if such study name makes sence
    study.optimize(lambda trial: objective(trial, num_reps), n_trials=num_trials)

    # Print best hyperparameters
    print("Best hyperparameters: ", study.best_params)


if __name__ == '__main__':
    num_trials = args.num_trials
    #TODO: BIG QUESTION: HOW DO WE UPDATE ENV?
    bayesian_optimization(num_trials, num_reps) 